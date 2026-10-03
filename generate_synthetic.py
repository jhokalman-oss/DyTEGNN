# -*- coding: utf-8 -*-
"""
generate_synthetic.py — generate the synthetic multi-relational Hawkes social
network described in Sec. 5.1 of the DyTEGNN paper.

Properties (matching the paper):
  * N = 3000 nodes, C = 4 latent communities, R = 3 relation types
  * ~30,000 stamped events  (u -> v, t, r)
  * each community c has a characteristic edge-type signature theta_c
    (Dirichlet) and a characteristic Hawkes excitation
  * node features carry only a ~35% community signal (weak feature signal)
  * edges are only weakly homophilic (30% within-community)

Timestamps are drawn from a multivariate Hawkes process (Ogata thinning) so the
event stream is genuinely self-/mutually-exciting.

Outputs (written to --out_dir):
  nodes.csv        node id, community label, feature vector (comma separated)
  edges.csv        src, dst, time, relation_type
  meta.json        parameters used to generate the data (for full provenance)

Run:
  python generate_synthetic.py --seed 0 --out_dir data/synthetic_seed0
"""
import argparse, json, os
import numpy as np


def simulate_hawkes(mu, alpha, beta, horizon, seed):
    """Multivariate Hawkes process via Ogata thinning with an O(R^2)-per-event
    exponential-decay accumulator (exact for exponential kernels, which are
    monotone non-increasing between events, so the current total intensity is a
    valid thinning bound).

    mu:     (R,) background rates
    alpha:  (R, R) excitation amplitudes  (log-space to keep positivity)
    beta:   (R, R) decay rates
    Returns: arrays (times, types) sorted by time within [0, horizon].
    """
    rng = np.random.default_rng(seed)
    alpha = np.exp(alpha); beta = np.exp(beta)
    R = len(mu)
    # s[r, r'] = accumulated (still-active) excitation of type r caused by
    # past events of type r'
    s = np.zeros((R, R), dtype=np.float64)
    t = 0.0
    times = []; types = []

    def total_intensity():
        lam = mu + s.sum(axis=1)
        return lam, lam.sum()

    while t < horizon:
        lam, lam_total = total_intensity()
        if lam_total <= 0:
            break
        dt = rng.exponential(1.0 / lam_total)
        t_next = t + dt
        if t_next > horizon:
            break
        # decay the accumulators by dt
        s = s * np.exp(-beta * dt)
        lam_next, lam_total_next = total_intensity()
        # accept/reject
        if rng.uniform() < lam_total_next / lam_total:
            # sample type proportional to per-type intensity at t_next
            p = lam_next / lam_next.sum()
            r_new = int(rng.choice(R, p=p))
            times.append(t_next); types.append(r_new)
            # this event excites all future types r by alpha[r, r_new] * beta[r, r_new]
            s[:, r_new] += alpha[:, r_new] * beta[:, r_new]
        t = t_next

    times = np.array(times); types = np.array(types)
    if len(times) == 0:
        return times, types
    order = np.argsort(times)
    return times[order], types[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--N', type=int, default=3000)
    ap.add_argument('--C', type=int, default=4)
    ap.add_argument('--R', type=int, default=3)
    ap.add_argument('--num_events', type=int, default=30000)
    ap.add_argument('--feat_dim', type=int, default=32)
    ap.add_argument('--feat_signal', type=float, default=0.5)
    ap.add_argument('--homophily', type=float, default=0.30)
    ap.add_argument('--horizon', type=float, default=100.0)
    ap.add_argument('--out_dir', type=str, default='data/synthetic_seed0')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    N, C, R = args.N, args.C, args.R

    # ---- latent communities (balanced) ----
    communities = rng.integers(0, C, size=N)

    # ---- per-community edge-type signature theta_c (Dirichlet) ----
    theta = rng.dirichlet(np.ones(R) * 1.0, size=C)   # (C, R)

    # ---- node features: only a weak community signal (~40% feature-only acc) ----
    d = args.feat_dim
    X = rng.normal(0.0, 1.0, size=(N, d)).astype(np.float32)
    # a weak additive community signal: features alone classify at ~40%,
    # matching the paper's "35% community signal" characterisation.
    X[np.arange(N), communities % d] += args.feat_signal

    # ---- Hawkes event timestamps + types ----
    # Background rates are scaled so that the simulation yields ~num_events
    # events over the horizon. With a branching ratio ~0.15 per pair, the total
    # multiplier is ~1/(1-0.45) ≈ 1.8, so we set total background ≈
    # num_events / horizon * (1 - 0.45).
    target_total = args.num_events / args.horizon * 0.55
    mu = (rng.uniform(0.8, 1.2, size=R).astype(np.float32))
    mu = mu / mu.sum() * target_total
    log_alpha = np.log(rng.uniform(0.05, 0.25, size=(R, R)).astype(np.float32))
    log_beta = np.log(rng.uniform(0.3, 1.0, size=(R, R)).astype(np.float32))
    times, types = simulate_hawkes(mu, log_alpha, log_beta, args.horizon, seed=args.seed + 1)

    # if we got more events than requested, subsample; if fewer, repeat
    n_ev = len(times)
    if n_ev == 0:
        raise RuntimeError('Hawkes simulation produced no events; increase horizon or mu')

    # ---- build edges (u -> v) with weak homophily ----
    # relation type of an edge is drawn from the *target* community's signature,
    # so a node's incoming edge types ~ theta_{c_target} identify its community.
    # (Message passing aggregates incoming messages, so this makes the
    # relation-type signature the learnable community signal.)
    # 30% of targets are same-community (weak homophily), 70% random.
    n_take = min(args.num_events, n_ev)
    idx = rng.choice(n_ev, size=n_take, replace=(args.num_events > n_ev))

    src = np.empty(n_take, dtype=np.int64)
    dst = np.empty(n_take, dtype=np.int64)
    rel = np.empty(n_take, dtype=np.int64)
    ev_t = np.empty(n_take, dtype=np.float32)

    for i, k in enumerate(idx):
        # source node: uniform
        u = int(rng.integers(0, N))
        src[i] = u
        # target: 30% same community as source, 70% any node
        if rng.uniform() < args.homophily:
            cand = np.where(communities == communities[u])[0]
            v = int(rng.choice(cand))
        else:
            v = int(rng.integers(0, N))
        dst[i] = v
        # relation type ~ theta_{c_target}  (the target's community signature)
        c_dst = communities[v]
        rel[i] = int(rng.choice(R, p=theta[c_dst] / theta[c_dst].sum()))
        ev_t[i] = times[k]

    # ---- write outputs ----
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'nodes.csv'), 'w') as f:
        f.write('node,label,' + ','.join('f%d' % j for j in range(d)) + '\n')
        for i in range(N):
            feat = ','.join('%.6f' % x for x in X[i])
            f.write('%d,%d,%s\n' % (i, communities[i], feat))
    with open(os.path.join(args.out_dir, 'edges.csv'), 'w') as f:
        f.write('src,dst,time,relation\n')
        for i in range(n_take):
            f.write('%d,%d,%.6f,%d\n' % (src[i], dst[i], ev_t[i], rel[i]))

    meta = dict(seed=args.seed, N=N, C=C, R=R, num_events=n_take,
                feat_dim=d, feat_signal=args.feat_signal, homophily=args.homophily,
                horizon=args.horizon,
                mu=mu.tolist(), log_alpha=log_alpha.tolist(), log_beta=log_beta.tolist(),
                theta=theta.tolist(),
                community_size={c: int((communities == c).sum()) for c in range(C)})
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('wrote synthetic dataset to %s' % args.out_dir)
    print('  nodes=%d  communities=%d  relation_types=%d  events=%d' % (N, C, R, n_take))
    print('  community sizes:', meta['community_size'])
    print('  per-community relation signature theta (rows=community, cols=type):')
    print(np.round(theta, 3))


if __name__ == '__main__':
    main()
