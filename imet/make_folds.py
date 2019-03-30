import argparse
from collections import defaultdict, Counter
import random
from pathlib import Path

import pandas as pd
import tqdm


data_root = Path('data')


def make_folds(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(data_root / 'train.csv')
    df['fold'] = -1
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        df.loc[item.Index, 'fold'] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_csv(data_root / 'folds.csv', index=None)
