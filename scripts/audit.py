"""Walk data/raw and build a per-image audit table."""
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # some real photos are huge; audit them anyway
RAW = Path("data/raw")
OUT = Path("reports/audit"); OUT.mkdir(parents=True, exist_ok=True)
SKIP = {".DS_Store", "Thumbs.db", "desktop.ini"}

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def inspect(path):
    info = dict(width=None, height=None, mode=None, format=None, readable=False, error="")
    try:
        with Image.open(path) as im:
            im.verify()                 # checks the file structure
        with Image.open(path) as im:    # verify() invalidates the object, so reopen
            im.load()                   # fully decodes pixels -> catches truncated files
            info.update(width=im.width, height=im.height, mode=im.mode,
                        format=im.format, readable=True)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info

rows = []
for class_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
    for path in sorted(class_dir.rglob("*")):
        if not path.is_file() or path.name in SKIP:
            continue
        rows.append(dict(label=class_dir.name, path=path.as_posix(),
                         file_size_kb=path.stat().st_size / 1024,
                         sha256=sha256_of(path), **inspect(path)))

df = pd.DataFrame(rows)
df.to_csv(OUT / "audit.csv", index=False)
ok = df[df.readable]

print("\n== Counts per class ==")
counts = df.groupby("label").agg(total=("path", "size"), readable=("readable", "sum"))
counts["corrupt"] = counts.total - counts.readable
print(counts)

print("\n== Width / height / file size (readable images) ==")
stats = ok.groupby("label")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"]).round(1)
print(stats)

dup_extra = ok.duplicated("sha256").sum()                 # copies beyond the first
dup_groups = (ok.sha256.value_counts() > 1).sum()
cross = (ok.groupby("sha256").label.nunique() > 1).sum()  # same bytes, both classes
print(f"\nCorrupt files: {(~df.readable).sum()}")
print(f"Duplicate groups: {dup_groups}, redundant copies: {dup_extra}, cross-class: {cross}")

print("\n== Format per class ==");      print(pd.crosstab(ok.label, ok.format))
print("\n== Colour mode per class =="); print(pd.crosstab(ok.label, ok["mode"]))
print("\nCorrupt file list:");          print(df.loc[~df.readable, ["path", "error"]].to_string())

counts.to_csv(OUT / "counts.csv"); stats.to_csv(OUT / "size_stats.csv")