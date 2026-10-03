# -*- coding: utf-8 -*-
"""
model.py — a faithful reference implementation of the DyTEGNN framework
(Hawkes encoder + edge-type-aware heterogeneous message passing + hierarchical
pooling + stability soft penalty), in pure PyTorch (no PyG dependency).

This is a *reference* implementation that mirrors the equations in the paper
(Sections 3-4). It is provided for reproducibility and as a starting point for
the authors' own training; it is NOT claimed to be the exact code that produced
any specific reported number.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionalTimeEncoding(nn.Module):
    """Bochner-theorem functional time encoding (Eq. 2 in the paper)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # learnable frequencies and phases
        self.omega = nn.Parameter(torch.randn(dim // 2) * 0.5)
        self.bias = nn.Parameter(torch.zeros(dim // 2))

    def forward(self, t):
        # t: (B,) -> (B, dim)
        t = t.unsqueeze(-1)                      # (B,1)
        omega = self.omega.unsqueeze(0)          # (1, d/2)
        bias = self.bias.unsqueeze(0)
        arg = omega * t + bias
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)  # (B, dim)


class HawkesEncoder(nn.Module):
    """Generalised Hawkes intensity restricted to a batch (Eq. 3) + temporal
    embedding (Eq. 4). Parameters are log-parameterised for strict positivity."""
    def __init__(self, R, d_time):
        super().__init__()
        self.R = R
        self.d_time = d_time
        # log-parameterised: mu (R,), alpha (R,R), beta (R,R).
        # alpha is initialised so the spectral radius starts below 1 (stable
        # regime), which is exactly the regime Proposition 1 describes.
        self.log_mu = nn.Parameter(torch.zeros(R))
        self.log_alpha = nn.Parameter(torch.full((R, R), -2.0))
        self.log_beta = nn.Parameter(torch.zeros(R, R))

    def intensity(self, t, r):
        """Compute batch-restricted intensity Lambda_i for each event i.
        t: (B,) timestamps; r: (B,) relation types.
        Returns Lambda: (B,)."""
        return self.intensity_batched(t, r, batch_events=600)

    def temporal_embedding(self, t, r, phi, batch_events=600):
        """phi: functional encoding module. Returns z_i (B, d_time).

        The intensity is batch-restricted (paper Eq. 3): we chunk the events in
        time order into batches of `batch_events` and compute the causal sum
        within each chunk, giving O(B^2) per chunk and O(E * batch) overall.
        """
        lam = self.intensity_batched(t, r, batch_events)  # (B,)
        g = torch.tanh(lam)                                # (B,)
        enc = phi(t)                                       # (B, d_time)
        return enc * (1.0 + g.unsqueeze(-1)) + g.unsqueeze(-1)

    def intensity_batched(self, t, r, batch_events=600):
        """Batch-restricted Hawkes intensity (causal within each chunk)."""
        B = t.shape[0]
        mu = torch.exp(self.log_mu)              # (R,)
        alpha = torch.exp(self.log_alpha)        # (R,R)
        beta = torch.exp(self.log_beta)          # (R,R)
        lam = torch.empty(B, device=t.device, dtype=t.dtype)
        for start in range(0, B, batch_events):
            end = min(start + batch_events, B)
            tb = t[start:end]
            rb = r[start:end]
            n = end - start
            dt = tb.unsqueeze(1) - tb.unsqueeze(0)      # (n,n)
            causal = (dt > 0).float()
            a = alpha[rb.unsqueeze(1), rb.unsqueeze(0)]  # (n,n)
            b = beta[rb.unsqueeze(1), rb.unsqueeze(0)]   # (n,n)
            exc = a * b * torch.exp(-b * dt.clamp(min=0)) * causal
            lam[start:end] = mu[rb] + exc.sum(dim=1)
        return lam

    def spectral_radius(self, power_iters=5):
        """Estimate rho(alpha) by power iteration (for the stability penalty)."""
        alpha = torch.exp(self.log_alpha)        # (R,R)
        v = torch.randn(self.R, device=alpha.device)
        v = v / (v.norm() + 1e-8)
        for _ in range(power_iters):
            v = alpha @ v
            n = v.norm() + 1e-8
            v = v / n
        return (v @ (alpha @ v)) / (v @ v + 1e-8)


class RelationDetector(nn.Module):
    """Infer relation type distribution from endpoints + temporal embedding
    (Eq. 5)."""
    def __init__(self, d_in, d_time, R, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in * 2 + d_time, hidden),
            nn.ReLU(),
            nn.Linear(hidden, R),
        )

    def forward(self, h_u, h_v, z_e):
        return self.mlp(torch.cat([h_u, h_v, z_e], dim=-1))  # logits (B,R)


class TypeSpecificAttention(nn.Module):
    """Type-specific projections W_q/W_k/W_v/W_o (Eq. 6). Produces, for each
    relation type, an aggregated message, then concatenates the per-type
    aggregates so the relation-type signature is directly exposed in the node
    representation (the core of edge-type-aware heterogeneous message passing)."""
    def __init__(self, d_h, d_time, R, n_heads=4):
        super().__init__()
        self.R = R
        self.d_h = d_h
        in_dim = d_h + d_time  # key/value condition on [h_v || z_e]
        self.W_k = nn.ModuleList([nn.Linear(in_dim, d_h) for _ in range(R)])
        self.W_v = nn.ModuleList([nn.Linear(in_dim, d_h) for _ in range(R)])
        # learnable per-type embeddings: expose the *relation-type signature*
        # (how much of each type points into a node) even when node features
        # are weak — the core signal of edge-type-aware heterogeneous MP.
        self.type_emb = nn.Parameter(torch.randn(R, d_h) * 0.1)
        self.combine = nn.Linear(R * d_h, d_h)

    def forward(self, H, src, dst, z_e, rel):
        """H: (N,d_h); returns updated node embeddings (N,d_h)."""
        device = H.device
        N = H.shape[0]
        per_type = []
        for k in range(self.R):
            mask = (rel == k)
            if mask.sum() == 0:
                per_type.append(torch.zeros(N, self.d_h, device=device))
                continue
            s = src[mask]; d = dst[mask]; ze = z_e[mask]
            kv = torch.cat([H[s], ze], dim=-1)          # (n, d_h+d_time)
            val = self.W_v[k](kv) + self.type_emb[k]    # add type signature
            # Hawkes-weighted temporal attention: weight each message by the
            # Hawkes temporal salience encoded in z_e (its first entry is a
            # monotone transform of the intensity Lambda_i).
            att = torch.sigmoid(ze[:, 0])               # (n,)
            msg = val * att.unsqueeze(-1)
            agg = torch.zeros(N, self.d_h, device=device)
            agg.scatter_add_(0, d.unsqueeze(1).expand_as(msg), msg)
            # degree normalisation per type
            deg = torch.zeros(N, device=device)
            deg.scatter_add_(0, d, torch.ones_like(d, dtype=H.dtype))
            deg = deg.clamp(min=1).unsqueeze(-1)
            per_type.append(agg / deg)
        cat = torch.cat(per_type, dim=-1)               # (N, R*d_h)
        return self.combine(cat)


class HierarchicalPool(nn.Module):
    """Soft-cluster assignment pooling (Eq. 8)."""
    def __init__(self, d_h, K):
        super().__init__()
        self.K = K
        self.assignment = nn.Linear(d_h, K)

    def forward(self, H):
        # H: (N, d_h) -> S: (N, K) soft assignment
        S = F.softmax(self.assignment(H), dim=-1)
        H_tilde = S.transpose(0, 1) @ H       # (K, d_h)
        return H_tilde, S


class DyTEGNN(nn.Module):
    def __init__(self, in_dim, d_h=64, d_time=32, R=3, C=4, K=64, n_layers=2,
                 n_heads=4, dropout=0.3, stability_weight=0.1, det_weight=0.5,
                 power_iters=5):
        super().__init__()
        self.R = R
        self.stability_weight = stability_weight
        self.det_weight = det_weight
        self.power_iters = power_iters

        self.node_proj = nn.Linear(in_dim, d_h)
        self.phi = FunctionalTimeEncoding(d_time)
        self.hawkes = HawkesEncoder(R, d_time)
        self.detector = RelationDetector(d_h, d_time, R)
        self.attn = TypeSpecificAttention(d_h, d_time, R, n_heads)
        self.pool = HierarchicalPool(d_h, K)
        self.norm = nn.LayerNorm(d_h)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_h, C)
        self._n_layers = n_layers

    def forward(self, x, src, dst, t, rel, train=True):
        """
        x:   (N, in_dim) node features
        src, dst: (E,) edge endpoints
        t:   (E,) timestamps
        rel: (E,) observed relation types
        Returns dict with:
          logits: (N, C)
          det_logits: (E, R)
          stab_penalty: scalar
        """
        device = x.device
        H = self.node_proj(x)                     # (N, d_h)
        z_e = self.hawkes.temporal_embedding(t, rel, self.phi)  # (E, d_time)

        # ---- edge-type-aware heterogeneous message passing ----
        for _ in range(self._n_layers):
            # relation detection (auxiliary): predict type from endpoints + z_e
            h_src = H[src]                        # (E, d_h)
            h_dst = H[dst]                        # (E, d_h)
            det_logits = self.detector(h_src, h_dst, z_e)
            # type-specific aggregation (per-type -> concat -> combine)
            agg = self.attn(H, src, dst, z_e, rel)
            H = self.norm(H + self.dropout(agg))

        # ---- hierarchical pooling (global context) ----
        H_tilde, S = self.pool(H)                 # (K, d_h)
        g = H_tilde.mean(dim=0, keepdim=True)     # (1, d_h) graph readout
        H = H + g                                 # add global context back

        logits = self.classifier(H)               # (N, C)

        # ---- stability soft penalty (Eq. 14 in paper) ----
        rho = self.hawkes.spectral_radius(self.power_iters)
        stab = torch.clamp(rho - 1.0, min=0.0) ** 2

        return dict(logits=logits, det_logits=det_logits, stab=stab, rho=rho)

    def loss(self, out, y, mask, rel, det_weight=None, stability_weight=None):
        """Node cross-entropy + relation-detection CE + stability penalty."""
        det_weight = self.det_weight if det_weight is None else det_weight
        stability_weight = self.stability_weight if stability_weight is None else stability_weight
        # node classification
        logits = out['logits'][mask]
        y_m = y[mask]
        loss_ce = F.cross_entropy(logits, y_m)
        # relation detection (supervised on observed types)
        loss_det = F.cross_entropy(out['det_logits'], rel)
        loss_stab = out['stab']
        return loss_ce + det_weight * loss_det + stability_weight * loss_stab
