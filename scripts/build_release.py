"""Build a deterministic ZIP from the verified file allowlist, never Git history."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from check_package import ROOT, check


def build(destination):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError('Release output must be outside the repository')
    if destination.exists():
        raise FileExistsError(destination)
    check()
    manifest = json.loads((ROOT / 'FILE_MANIFEST.json').read_text())
    files = sorted([*manifest['files'], 'FILE_MANIFEST.json'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in files:
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 20, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (ROOT / name).read_bytes())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f'{digest}  {destination.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    build(parser.parse_args().output)
