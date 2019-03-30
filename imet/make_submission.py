import argparse

import pandas as pd

from .utils import mean_df


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('output')
    arg('threshold', type=float, default=0.2)
    args = parser.parse_args()
    # TODO probably makes sense to use make_submission from amazon
    sample_submission = pd.read_csv(
        'data/sample_submission.csv', index_col='id')
    dfs = []
    for prediction in args.predictions:
        df = pd.read_hdf(prediction, index_col='id')
        df = df.reindex(sample_submission.index)
        dfs.append(df)
    df = pd.concat(dfs)
    df = mean_df(df)
    df = df > args.threshold  # FIXME check best approach
    df = df.apply(get_classes, axis=1)
    df.name = 'Predicted'
    df.to_csv(args.output, header=True)


def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)
