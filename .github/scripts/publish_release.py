#!/usr/bin/env python3
"""Idempotently publish verified release assets to a PyPI-compatible index."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from email.parser import Parser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

NORMALIZE_NAME_RE = re.compile(r"[-_.]+")


def _origin(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    return parsed.scheme.lower(), parsed.netloc.lower()


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Request | None:
        redirected = super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )
        if redirected is not None and _origin(request.full_url) != _origin(new_url):
            redirected.remove_header("Authorization")
        return redirected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_name(name: str) -> str:
    return NORMALIZE_NAME_RE.sub("-", name).lower()


def _project_name(filename: str, version: str) -> str:
    if filename.endswith(".whl"):
        distribution, parsed_version, _, _ = parse_wheel_filename(filename)
    elif filename.endswith(".tar.gz"):
        distribution, parsed_version = parse_sdist_filename(filename)
    else:
        raise ValueError(f"unsupported distribution filename: {filename!r}")
    if parsed_version != Version(version):
        raise ValueError(
            f"artifact {filename!r} has version {parsed_version}, "
            f"not manifest version {version}"
        )
    return _canonicalize_name(distribution)


def _authorization(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def _read_url(
    url: str,
    *,
    repository_origin: tuple[str, str],
    username: str,
    password: str,
) -> tuple[bytes, str]:
    headers = {"Accept": "text/html, application/octet-stream"}
    if _origin(url) == repository_origin:
        headers["Authorization"] = _authorization(username, password)
    request = Request(url, headers=headers)
    with build_opener(_SafeRedirectHandler).open(request, timeout=30) as response:
        return response.read(), response.geturl()


def _remote_digest(
    *,
    simple_url: str,
    project: str,
    filename: str,
    username: str,
    password: str,
) -> str | None:
    project_url = urljoin(simple_url.rstrip("/") + "/", f"{project}/")
    repository_origin = _origin(project_url)
    try:
        page, resolved_project_url = _read_url(
            project_url,
            repository_origin=repository_origin,
            username=username,
            password=password,
        )
    except HTTPError as error:
        if error.code == 404:
            return None
        raise

    parser = _LinkParser()
    parser.feed(page.decode("utf-8"))
    matching: list[str] = []
    for href in parser.links:
        asset_url = urljoin(resolved_project_url, href)
        if unquote(Path(urlparse(asset_url).path).name) == filename:
            matching.append(asset_url)
    if not matching:
        return None
    if len(matching) != 1:
        raise ValueError(f"repository returned duplicate links for {filename}")

    asset_url = matching[0]
    fragment = urlparse(asset_url).fragment
    if fragment.startswith("sha256="):
        digest = fragment.removeprefix("sha256=").lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest):
            return digest

    content, _ = _read_url(
        asset_url,
        repository_origin=repository_origin,
        username=username,
        password=password,
    )
    return hashlib.sha256(content).hexdigest()


def _upload(
    *,
    path: Path,
    repository_url: str,
    username: str,
    password: str,
) -> None:
    environment = {
        **os.environ,
        "TWINE_USERNAME": username,
        "TWINE_PASSWORD": password,
    }
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("twine_upload.py")),
            "upload",
            "--non-interactive",
            "--repository-url",
            repository_url,
            str(path),
        ],
        check=True,
        env=environment,
    )


def _wheel_dependencies(path: Path) -> set[str]:
    with ZipFile(path) as wheel:
        metadata_paths = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(f"{path.name} must contain exactly one METADATA file")
        metadata = Parser().parsestr(wheel.read(metadata_paths[0]).decode("utf-8"))
    return {
        _canonicalize_name(Requirement(value).name)
        for value in metadata.get_all("Requires-Dist", [])
    }


def _publication_order(
    *,
    asset_dir: Path,
    files: list[dict[str, object]],
    version: str,
) -> list[str]:
    projects: dict[str, list[str]] = {}
    for entry in files:
        filename = entry["name"]
        if not isinstance(filename, str):
            raise ValueError("invalid release manifest filename")
        project = _project_name(filename, version)
        projects.setdefault(project, []).append(filename)

    dependencies: dict[str, set[str]] = {project: set() for project in projects}
    if len(projects) > 1:
        for project, filenames in projects.items():
            for filename in filenames:
                if filename.endswith(".whl"):
                    dependencies[project].update(
                        dependency
                        for dependency in _wheel_dependencies(asset_dir / filename)
                        if dependency in projects and dependency != project
                    )

    remaining = set(projects)
    ordered_projects: list[str] = []
    while remaining:
        ready = sorted(
            project for project in remaining if not (dependencies[project] & remaining)
        )
        if not ready:
            detail = {
                project: sorted(dependencies[project] & remaining)
                for project in sorted(remaining)
            }
            raise ValueError(f"internal distribution dependency cycle: {detail}")
        ordered_projects.extend(ready)
        remaining.difference_update(ready)

    return [
        filename
        for project in ordered_projects
        for filename in sorted(
            projects[project],
            key=lambda name: (not name.endswith(".tar.gz"), name),
        )
    ]


def publish_assets(
    *,
    asset_dir: Path,
    manifest: dict[str, object],
    repository_url: str,
    simple_url: str,
    username: str,
    password: str,
) -> list[tuple[str, str]]:
    """Upload missing files and accept existing files only when hashes match."""
    version = manifest.get("version")
    files = manifest.get("files")
    if not isinstance(version, str) or not isinstance(files, list):
        raise ValueError("invalid release manifest")

    candidates: dict[str, tuple[Path, str, str | None]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("invalid release manifest file entry")
        filename = entry.get("name")
        expected_digest = entry.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected_digest, str):
            raise ValueError("invalid release manifest file metadata")
        path = asset_dir / filename
        if _sha256(path) != expected_digest:
            raise ValueError(f"local release asset digest mismatch: {filename}")

        project = _project_name(filename, version)
        remote_digest = _remote_digest(
            simple_url=simple_url,
            project=project,
            filename=filename,
            username=username,
            password=password,
        )
        if remote_digest is not None and remote_digest != expected_digest:
            raise ValueError(
                f"repository already contains {filename} with a different digest"
            )
        candidates[filename] = (path, project, remote_digest)

    # Finish every local and remote check before the first irreversible upload,
    # then expose foundations before their exact-pinned dependents.
    order = _publication_order(asset_dir=asset_dir, files=files, version=version)
    results: list[tuple[str, str]] = []
    for filename in order:
        path, project, remote_digest = candidates[filename]
        expected_digest = _sha256(path)
        if remote_digest is not None:
            results.append((filename, "already published"))
            continue

        try:
            _upload(
                path=path,
                repository_url=repository_url,
                username=username,
                password=password,
            )
        except subprocess.CalledProcessError:
            # The server may have stored the file while the client lost the
            # response. Re-query briefly before declaring the release unsafe.
            for attempt in range(5):
                if attempt:
                    time.sleep(2)
                remote_digest = _remote_digest(
                    simple_url=simple_url,
                    project=project,
                    filename=filename,
                    username=username,
                    password=password,
                )
                if remote_digest is not None:
                    break
            if remote_digest != expected_digest:
                raise
            results.append((filename, "published before retry"))
        else:
            results.append((filename, "published"))
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--simple-url", required=True)
    parser.add_argument("--username", default=os.environ.get("TWINE_USERNAME", ""))
    parser.add_argument("--password", default=os.environ.get("TWINE_PASSWORD", ""))
    return parser


def main() -> int:
    """Publish all files described by a previously verified manifest."""
    args = _build_parser().parse_args()
    if not args.username or not args.password:
        raise SystemExit(
            "TWINE_USERNAME and TWINE_PASSWORD (or matching arguments) are required"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = publish_assets(
        asset_dir=args.asset_dir,
        manifest=manifest,
        repository_url=args.repository_url,
        simple_url=args.simple_url,
        username=args.username,
        password=args.password,
    )
    for filename, state in results:
        print(f"{filename}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
