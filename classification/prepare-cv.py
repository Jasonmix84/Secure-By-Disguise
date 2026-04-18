
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os,sys

# 1. Setup path
root = Path(sys.argv[1])

# 2. Extract paths and class names (parent folder name)
# This looks for any file inside subfolders of my_dataset
data = []
for file_path in root.rglob("*"):
    if file_path.is_file():
        data.append({
            "path": str(file_path),
            "class_name": file_path.parent.name
        })

df = pd.DataFrame(data)

# 3. Encode class names to integers (e.g., "cat" -> 0, "dog" -> 1)
le = LabelEncoder()
df['label'] = le.fit_transform(df['class_name'])

# 4. Generate Stratified Folds
# We shuffle because files are usually read in order by class
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = -1

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df['path'], df['label'])):
    df.loc[val_idx, 'fold'] = fold_idx

# 5. Save the master map
df.to_csv(os.path.join(sys.argv[2]), index=False)

# Optional: Print mapping for your records
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Class Mapping: {mapping}")
print(df.head())
