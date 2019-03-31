import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam
import tqdm

from . import models
from .dataset import TrainDataset, TTADataset, get_ids, N_CLASSES, DATA_ROOT
from .transforms import train_transform, test_transform
from .utils import (
    write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
    ON_KAGGLE)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict_valid', 'predict_test'])
    arg('run_root')
    arg('--model', default='resnet50')
    arg('--pretrained', type=int, default=1)
    arg('--batch-size', type=int, default=64)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--lr', type=float, default=1e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=100)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=4)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    folds = pd.read_csv('folds.csv')
    train_root = DATA_ROOT / ('train_sample' if args.use_sample else 'train')
    if args.use_sample:
        folds = folds[folds['Id'].isin(set(get_ids(train_root)))]
    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    model = getattr(models, args.model)(
        num_classes=N_CLASSES, pretrained=args.pretrained)
    use_cuda = cuda.is_available()
    fresh_params = list(model.fresh_params())
    all_params = list(model.parameters())
    if use_cuda:
        model = model.cuda()

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(train_fold, train_transform)
        valid_loader = make_loader(valid_fold, test_transform)
        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda params, lr: Adam(params, lr),
            use_cuda=use_cuda,
        )

        if args.pretrained:
            if train(params=fresh_params, n_epochs=1, **train_kwargs):
                train(params=all_params, **train_kwargs)
        else:
            train(params=all_params, **train_kwargs)

    elif args.mode == 'validate':
        valid_loader = make_loader(valid_fold, test_transform)
        load_model(model, run_root / 'model.pt')
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                   use_cuda=use_cuda)

    elif args.mode.startswith('predict'):
        load_model(model, run_root / 'best-model.pt')
        predict_kwargs = dict(
            batch_size=args.batch_size,
            tta=args.tta,
            use_cuda=use_cuda,
            workers=args.workers,
        )
        if args.mode == 'predict_valid':
            predict(model, df=valid_fold, root=train_root,
                    out_path=run_root / 'val.h5',
                    **predict_kwargs)
        elif args.mode == 'predict_test':
            test_root = DATA_ROOT / (
                'test_sample' if args.use_sample else 'test')
            ss = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
            if args.use_sample:
                ss = ss[ss['id'].isin(set(get_ids(test_root)))]
            if args.limit:
                ss = ss[:args.limit]
            predict(model, df=ss, root=test_root,
                    out_path=run_root / 'test.h5',
                    **predict_kwargs)


def predict(model, root: Path, df: pd.DataFrame, out_path: Path,
            batch_size: int, tta: int, workers: int, use_cuda: bool):
    loader = DataLoader(
        dataset=TTADataset(root, df, test_transform, tta=tta),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    all_outputs, all_ids = [], []
    with torch.no_grad():
        for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
            if use_cuda:
                inputs = inputs.cuda()
            outputs = torch.sigmoid(model(inputs))
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')


def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=2) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)
    optimizer = init_optimizer(params, lr)

    run_root = Path(args.run_root)
    model_path = run_root / 'model.pt'
    best_model_path = run_root / 'best-model.pt'
    if model_path.exists():
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = _reduce_loss(criterion(outputs, targets))
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}')
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, use_cuda)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr_changes +=1
                if lr_changes > max_lr_changes:
                    break
                lr /= 5
                print(f'lr updated to {lr}')
                lr_reset_epoch = epoch
                optimizer = init_optimizer(params, lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return False
    return True


def validation(
        model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_losses.append(_reduce_loss(loss).item())
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                all_targets, y_pred, beta=2, average='samples')

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    return metrics


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


if __name__ == '__main__':
    main()
