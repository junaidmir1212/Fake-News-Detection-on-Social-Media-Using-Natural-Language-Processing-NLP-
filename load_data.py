import pandas as pd
import os

# ── Kaggle Dataset ───────────────────────────────────────────────────────────
fake_df = pd.read_csv('data/Kaggle Fake News Dataset/Fake.csv')
true_df = pd.read_csv('data/Kaggle Fake News Dataset/True.csv')

fake_df['label'] = 0
true_df['label'] = 1

kaggle_df = pd.concat([fake_df, true_df], ignore_index=True)
kaggle_df = kaggle_df[['text', 'label']].dropna(subset=['text'])
print(f"Kaggle dataset          : {len(kaggle_df):,} rows")

# ── Mendeley Dataset — 4 sub-datasets ───────────────────────────────────────
mendeley_base = 'data/Mendeley Fake News Dataset'

sub_datasets = [
    'ISOT Fake News Dataset',
    'Fake News Dataset',
    'Fake or Real News Dataset',
    'Fake News Detection Dataset',
]

mendeley_frames = []

for folder in sub_datasets:
    train_path = os.path.join(mendeley_base, folder, 'train.csv')
    test_path  = os.path.join(mendeley_base, folder, 'test.csv')

    df_train = pd.read_csv(train_path, sep=';')
    df_test  = pd.read_csv(test_path,  sep=';')

    mendeley_frames.append(df_train)
    mendeley_frames.append(df_test)

    count = len(df_train) + len(df_test)
    print(f"  {folder:<35}: {count:,} rows")

mendeley_df = pd.concat(mendeley_frames, ignore_index=True)
mendeley_df = mendeley_df[['text', 'label']].dropna(subset=['text'])
print(f"Mendeley total          : {len(mendeley_df):,} rows")

# ── Merge both ───────────────────────────────────────────────────────────────
df = pd.concat([kaggle_df, mendeley_df], ignore_index=True)
df = df.drop_duplicates(subset=['text'])
df = df.reset_index(drop=True)

print(f"\nCombined total          : {len(df):,} rows")
print(f"Fake news  (label=0)    : {len(df[df['label']==0]):,}")
print(f"Real news  (label=1)    : {len(df[df['label']==1]):,}")
print(f"\nSample:")
print(df.head())

# Save
df.to_csv('data/news_dataset.csv', index=False)
print("\nDataset saved to data/news_dataset.csv")