import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from visionguard.imaging import normalize, JPEG_QUALITY

Image.MAX_IMAGE_PIXELS = None
LABEL_MAP = {"AiArtData": "ai", "RealArt": "real"}   # <- edit to YOUR folder names
SEED = 42

df = pd.read_csv("reports/audit/audit.csv")
df = df[df.readable].copy()
df["label"] = df["label"].map(LABEL_MAP)
assert df.label.notna().all(), "Unmapped folder name - fix LABEL_MAP"

conflict = df.groupby("sha256").label.nunique()
df = df[~df.sha256.isin(conflict[conflict > 1].index)]   # drop label-conflicting images
df = df.drop_duplicates("sha256", keep="first")          # keep one copy of each duplicate

records = []
for _, r in df.iterrows():
    dst = Path("data/clean") / r.label / f"{r.sha256[:16]}.jpg"   # hash name = safe & unique
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(r.path) as im:
        normalize(im).save(dst, "JPEG", quality=JPEG_QUALITY)
    records.append(dict(filepath=dst.as_posix(), label=r.label,
                        y=int(r.label == "ai"), source=r.path))

clean = pd.DataFrame(records)
train, tmp = train_test_split(clean, test_size=0.30, stratify=clean.y, random_state=SEED)
val, test = train_test_split(tmp, test_size=0.50, stratify=tmp.y, random_state=SEED)

Path("data/splits").mkdir(parents=True, exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(f"data/splits/{name}.csv", index=False)
    print(f"{name:5}: {len(part):4d} images, AI share = {part.y.mean():.2f}")