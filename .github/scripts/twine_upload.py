"""Upload distributions with Twine and JetBrains Packages' legacy MD5 field."""

import hashlib
import sys
from pathlib import Path
from typing import Any

from twine.cli import dispatch
from twine.package import PackageFile


_metadata_dictionary = PackageFile.metadata_dictionary


def metadata_dictionary_with_md5(self: PackageFile) -> dict[str, Any]:
    metadata = _metadata_dictionary(self)
    with Path(self.filename).open("rb") as distribution:
        metadata["md5_digest"] = hashlib.file_digest(distribution, "md5").hexdigest()
    return metadata


PackageFile.metadata_dictionary = metadata_dictionary_with_md5


if __name__ == "__main__":
    dispatch(sys.argv[1:])
