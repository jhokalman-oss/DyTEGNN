# -*- coding: utf-8 -*-
"""
train_eval.py — train DyTEGNN on a generated synthetic dataset and evaluate
node classification (macro-F1, accuracy, AUC), logging per-epoch metrics to JSON
and CSV for full reproducibility.

Run:
  python train_eval.py --data_dir data/synthetic_seed0 --seed 0 --epochs 200 \
      --out_dir runs/synthetic_seed0
"""
import argparse, csv, json, os
import numpy as np
import torch

from model import DyTEGNN


def load_dataset(data_dir):
    nodes = []
    labels = []
    with open(os.path.join(data_dir, 'nodes.csv')) as f:
        header = f.readline().strip().split(',')
        feat_cols = header[2:]
        for line in f:
            parts = line.strip().split(',')
            node = int(parts[0]); lab = int(parts[1])
            feat = [float(v) for v in parts[2:]]
            nodes.append(node); labels.append(lab)
            # store features in node order (assumed sorted by node id)
    # re-read features in node order
    node_feats = {}
    node_labs = {}
    with open(os.path.join(data_dir, 'nodes.csv')) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            node_feats[int(parts[0])] = [float(v) for v in parts[2:]]
            node_labs[int(parts[0])] = int(parts[1])
    N = max(node_feats) + 1
    d = len(next(iter(node_feats.values())))
    X = np.zeros((N, d), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    for n in node_feats:
        X[n] = node_feats[n]; y[n] = node_labs[n]

    src = []; dst = []; times = []; rel = []
    with open(os.path.join(data_dir, 'edges.csv')) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            src.append(int(parts[0])); dst.append(int(parts[1]))
            times.append(float(parts[2])); rel.append(int(parts[3]))
    E = len(src)
    src = np.array(src, dtype=np.int64); dst = np.array(dst, dtype=np.int64)
    times = np.array(times, dtype=np.float32); rel = np.array(rel, dtype=np.int64)
    return X, y, src, dst, times, rel


def macro_f1(pred, truth, C):
    f1s = []
    for c in range(C):
        tp = ((pred == c) & (truth == c)).sum()
        fp = ((pred == c) & (truth != c)).sum()
        fn = ((pred != c) & (truth == c)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


def auc_from_scores(scores, truth):
    # scores: (N,C) softmax; truth: (N,) — one-vs-rest averaged AUC via ranks
    C = scores.shape[1]
    aucs = []
    for c in range(C):
        s = scores[:, c]
        pos = s[truth == c]; neg = s[truth != c]
        if len(pos) == 0 or len(neg) == 0:
            aucs.append(0.5); continue
        # Mann-Whitney
        n_pos, n_neg = len(pos), len(neg)
        ranks = np.concatenate([pos, neg])
        order = np.argsort(ranks)
        ranks_sorted = np.empty_like(order)
        ranks_sorted[order] = np.arange(1, len(ranks) + 1)
        sum_ranks_pos = ranks_sorted[:n_pos].sum()
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
        aucs.append(u / (n_pos * n_neg))
    return float(np.mean(aucs))


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'

    X, y, src, dst, times, rel = load_dataset(args.data_dir)
    N, d = X.shape
    C = int(y.max()) + 1
    R = int(rel.max()) + 1
    print('dataset: N=%d d=%d C=%d R=%d E=%d' % (N, d, C, R, len(src)))
    print('device:', device)

    # temporal split: train/val/test 60/20/20 by edge time (for label split we
    # keep all nodes but split is informational for the link task; here we use a
    # random 60/20/20 node split for classification).
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_tr = int(0.6 * N); n_va = int(0.2 * N)
    train_idx = idx[:n_tr]; val_idx = idx[n_tr:n_tr+n_va]; test_idx = idx[n_tr+n_va:]
    train_mask = np.zeros(N, dtype=bool); train_mask[train_idx] = True
    val_mask = np.zeros(N, dtype=bool); val_mask[val_idx] = True
    test_mask = np.zeros(N, dtype=bool); test_mask[test_idx] = True

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)
    src_t = torch.tensor(src, device=device)
    dst_t = torch.tensor(dst, device=device)
    t_t = torch.tensor(times, device=device)
    rel_t = torch.tensor(rel, device=device)
    train_mask_t = torch.tensor(train_mask, device=device)
    val_mask_t = torch.tensor(val_mask, device=device)
    test_mask_t = torch.tensor(test_mask, device=device)

    model = DyTEGNN(in_dim=d, d_h=args.d_h, d_time=args.d_time, R=R, C=C,
                    K=args.K, n_layers=args.n_layers, n_heads=args.n_heads,
                    dropout=args.dropout, stability_weight=args.stab_w,
                    det_weight=args.det_w, power_iters=args.power_iters).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'log.csv')
    flds = ['epoch', 'train_loss', 'val_macro_f1', 'val_acc', 'val_auc', 'rho']
    logf = open(log_path, 'w', newline='')
    writer = csv.DictWriter(logf, fieldnames=flds)
    writer.writeheader()

    best_val = 0.0; best_state = None; best_epoch = 0
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(X_t, src_t, dst_t, t_t, rel_t, train=True)
        loss = model.loss(out, y_t, train_mask_t, rel_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_v = model(X_t, src_t, dst_t, t_t, rel_t, train=False)
            logits = out_v['logits']
            pred = logits.argmax(dim=-1)
            scores = torch.softmax(logits, dim=-1).cpu().numpy()
            val_mf1 = macro_f1(pred[val_mask_t].cpu().numpy(), y_t[val_mask_t].cpu().numpy(), C)
            val_acc = float((pred[val_mask_t] == y_t[val_mask_t]).float().mean())
            val_auc = auc_from_scores(scores[val_idx], y[val_idx])
            rho = float(out_v['rho'].item())

        writer.writerow(dict(epoch=ep, train_loss=float(loss.item()),
                             val_macro_f1=val_mf1, val_acc=val_acc, val_auc=val_auc,
                             rho=rho))
        if val_mf1 > best_val:
            best_val = val_mf1; best_epoch = ep; patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep % 20 == 0 or ep == args.epochs:
            print('epoch %4d  loss=%.4f  val_macro_f1=%.4f  val_acc=%.4f  val_auc=%.4f  rho=%.4f'
                  % (ep, loss.item(), val_mf1, val_acc, val_auc, rho))
        if patience >= args.patience:
            print('early stop at epoch %d' % ep)
            break

    logf.close()

    # final test evaluation on best state
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(X_t, src_t, dst_t, t_t, rel_t, train=False)
        logits = out['logits']
        pred = logits.argmax(dim=-1)
        scores = torch.softmax(logits, dim=-1).cpu().numpy()
        test_mf1 = macro_f1(pred[test_mask_t].cpu().numpy(), y_t[test_mask_t].cpu().numpy(), C)
        test_acc = float((pred[test_mask_t] == y_t[test_mask_t]).float().mean())
        test_auc = auc_from_scores(scores[test_idx], y[test_idx])
        rho = float(out['rho'].item())

    summary = dict(seed=args.seed, data_dir=args.data_dir,
                   d_h=args.d_h, d_time=args.d_time, R=int(R), C=int(C), K=args.K,
                   n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout,
                   lr=args.lr, wd=args.wd, stab_w=args.stab_w, det_w=args.det_w,
                   power_iters=args.power_iters, epochs=ep, best_epoch=best_epoch,
                   best_val_macro_f1=best_val,
                   test_macro_f1=test_mf1, test_acc=test_acc, test_auc=test_auc,
                   rho=rho, device=device)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    torch.save(best_state, os.path.join(args.out_dir, 'model.pt'))

    print('--- summary ---')
    for k, v in summary.items():
        print('  %s = %s' % (k, v))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data/synthetic_seed0')
    ap.add_argument('--out_dir', type=str, default='runs/synthetic_seed0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--d_h', type=int, default=64)
    ap.add_argument('--d_time', type=int, default=32)
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--stab_w', type=float, default=0.1)
    ap.add_argument('--det_w', type=float, default=0.5)
    ap.add_argument('--power_iters', type=int, default=5)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    train(args)
