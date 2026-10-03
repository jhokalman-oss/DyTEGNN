# -*- coding: utf-8 -*-
"""
run_all_seeds.py — generate the synthetic dataset and train DyTEGNN over all five
TGB-recommended seeds, then aggregate results (mean ± std) into
results/synthetic_summary.json.

Run:  python run_all_seeds.py
"""
import json, os, subprocess, sys
import numpy as np

PY = sys.executable
SEEDS = [0, 1, 2, 3, 4]

def run(cmd):
    print('+ ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    summaries = []
    for s in SEEDS:
        data_dir = os.path.join(base, 'data', 'synthetic_seed%d' % s)
        run_dir = os.path.join(base, 'runs', 'synthetic_seed%d' % s)
        run([PY, 'generate_synthetic.py', '--seed', str(s), '--out_dir', data_dir])
        run([PY, 'train_eval.py', '--data_dir', data_dir, '--seed', str(s),
             '--out_dir', run_dir])
        with open(os.path.join(run_dir, 'summary.json')) as f:
            summaries.append(json.load(f))

    mf1 = [x['test_macro_f1'] for x in summaries]
    acc = [x['test_acc'] for x in summaries]
    auc = [x['test_auc'] for x in summaries]
    rho = [x['rho'] for x in summaries]

    out = {
        'seeds': SEEDS,
        'macro_f1_mean': float(np.mean(mf1)), 'macro_f1_std': float(np.std(mf1)),
        'acc_mean': float(np.mean(acc)), 'acc_std': float(np.std(acc)),
        'auc_mean': float(np.mean(auc)), 'auc_std': float(np.std(auc)),
        'rho_mean': float(np.mean(rho)), 'rho_std': float(np.std(rho)),
        'per_seed': summaries,
    }
    os.makedirs(os.path.join(base, 'results'), exist_ok=True)
    with open(os.path.join(base, 'results', 'synthetic_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\n=== AGGREGATED (mean ± std over 5 seeds) ===')
    print('macro-F1: %.4f ± %.4f' % (out['macro_f1_mean'], out['macro_f1_std']))
    print('accuracy: %.4f ± %.4f' % (out['acc_mean'], out['acc_std']))
    print('AUC:      %.4f ± %.4f' % (out['auc_mean'], out['auc_std']))
    print('rho:      %.4f ± %.4f' % (out['rho_mean'], out['rho_std']))
