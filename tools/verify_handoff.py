"""Verify all retained handoff files without third-party dependencies."""
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
manifest = root / 'MANIFEST.sha256'
failed = []
count = 0
for line in manifest.read_text(encoding='utf-8').splitlines():
    expected, name = line.split('  ', 1)
    path = root / name
    if not path.is_file():
        failed.append(f'MISSING {name}')
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        failed.append(f'MISMATCH {name}')
    count += 1
for failure in failed:
    print(failure)
print(f'Checked {count} files; failures: {len(failed)}')
sys.exit(bool(failed))
