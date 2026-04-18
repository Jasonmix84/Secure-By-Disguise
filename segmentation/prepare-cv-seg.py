import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold


root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else 5

images_dir = root / "images"
labels_dir = root / "labels"

# Build dataframe directly with paths
files = sorted(p.name for p in images_dir.iterdir() if p.is_file())

df = pd.DataFrame({
    "file_name": files,
    "image_path": [str(images_dir / f) for f in files],
    "label_path": [str(labels_dir / f) for f in files],
    "fold": -1,
})

# Assign fold indices
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (_, test_idx) in enumerate(kf.split(df)):
    df.loc[test_idx, "fold"] = fold

# Save
df.to_csv(out_path, index=False)

print(df.head())
print(df["fold"].value_counts().sort_index())
