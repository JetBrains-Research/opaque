"""Upload distributions with Twine and JetBrains Packages' legacy MD5 field."""

import hashlib
import sys
from pathlib import Path
from typing import Any

from twine.cli import dispatch
from twine.package import PackageFile


_metadata_dictionary = PackageFile.metadata_dictionary


def md5_digest(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as distribution:
        for chunk in iter(lambda: distribution.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_dictionary_with_md5(self: PackageFile) -> dict[str, Any]:
    metadata = _metadata_dictionary(self)
    metadata["md5_digest"] = md5_digest(Path(self.filename))
    return metadata


PackageFile.metadata_dictionary = metadata_dictionary_with_md5


if __name__ == "__main__":
    dispatch(sys.argv[1:])
