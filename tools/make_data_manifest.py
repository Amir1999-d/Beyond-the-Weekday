"""
make_data_manifest.py — record or verify SHA-256 checksums of the input data.

The GTFS and ISTAT inputs are not redistributed with this repository, and
archived GTFS feeds are hard to recover once the operator replaces them. The
manifest lets anyone confirm that their local copies are byte-identical to the
files used for the thesis run.

    python tools/make_data_manifest.py           # write data/MANIFEST.sha256
    python tools/make_data_manifest.py --check   # verify local files against it

Operator (ATM) files are never listed. Run from the repository root.
"""
import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MANIFEST = os.path.join(DATA, "MANIFEST.sha256")
SKIP_NAMES = {"README.md", "MANIFEST.sha256", ".DS_Store", "Thumbs.db", "desktop.ini"}
OPERATOR_MARKERS = ("tornellati", "tesista")          # private ATM workbook


def is_operator_file(rel_path):
    low = rel_path.lower()
    return any(m in low for m in OPERATOR_MARKERS)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_files():
    out = []
    for dirpath, _, files in os.walk(DATA):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, DATA).replace(os.sep, "/")
            if name in SKIP_NAMES or is_operator_file(rel):
                continue
            out.append(rel)
    return sorted(out)


def write():
    files = data_files()
    if not files:
        sys.exit(f"No data files found under {DATA}")
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        for rel in files:
            f.write(f"{sha256(os.path.join(DATA, rel))}  {rel}\n")
    print(f"Wrote {len(files)} checksums to {os.path.relpath(MANIFEST, ROOT)}")


def check():
    if not os.path.exists(MANIFEST):
        sys.exit(f"{os.path.relpath(MANIFEST, ROOT)} not found")
    bad = 0
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            digest, rel = line.rstrip("\n").split("  ", 1)
            path = os.path.join(DATA, rel)
            if not os.path.exists(path):
                print(f"MISSING  {rel}"); bad += 1
            elif sha256(path) != digest:
                print(f"DIFFERS  {rel}"); bad += 1
    print("All files match the manifest." if not bad else f"{bad} file(s) missing or different.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify instead of writing")
    check() if ap.parse_args().check else write()
