"""
models/temporal_gnn.py

Temporal GNN baseline: a snapshot-TGN variant.

Rationale
---------
The original TGN (Rossi et al. 2020) is designed for streaming event data
where every interaction carries a continuous timestamp and a memory state
is updated per event. Our three target datasets use a coarser temporal
structure — node-level time buckets derived from edge timestamps — so a
per-event memory module is overkill and would require a streaming data
pipeline the rest of the repo does not have.

Instead we implement **SnapshotTGN**: a discrete-time approximation of TGN
that keeps the three architectural pieces reviewers expect to see —

  * per-node memory updated once per snapshot with a GRUCell
  * a learned time-encoder φ(Δt) (the sinusoidal encoder from Xu et al.
    2020) concatenated with the current message
  * a graph attention module for the final per-node readout

... while remaining a drop-in, single-forward-pass model usable by the
existing inductive / transductive harness. For a faithful streaming TGN
runner see ``experiments/run_tgn_stream.py`` (not included — requires
event-level timestamps DGraphFin does not provide).

Forward signature matches the other models: ``(x, edge_index, edge_attr)``.
Snapshots are recovered internally from ``time_step`` when passed through
the ``FraudDataset`` container, which is how the experiment harness calls
this model.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv
from torch_geometric.utils import subgraph


class TimeEncoder(nn.Module):
    """Sinusoidal time encoder (Xu et al., 2020).

    Maps a scalar Δt to a ``dim``-D vector via learned frequencies. Used by
    the memory module to condition the update on how long it has been since
    a node was last observed.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim  = dim
        self.freq = nn.Parameter(
            torch.from_numpy(
                (1.0 / 10 ** (2 * (torch.arange(dim).float() // 2) / dim)).numpy()
            ).float()
        )
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: [N]  → [N, dim].  We compute sin/cos via interleaving of
        # concatenated halves rather than in-place slice assignment so the
        # op is out-of-place and safe under autograd.
        z = dt.unsqueeze(-1) * self.freq + self.phase
        even = torch.sin(z[..., 0::2])
        odd  = torch.cos(z[..., 1::2])
        # Interleave: [sin_0, cos_0, sin_1, cos_1, ...] → shape [N, dim]
        stacked = torch.stack([even, odd], dim=-1)
        return stacked.reshape(*z.shape[:-1], z.shape[-1])


class SnapshotTGN(nn.Module):
    """Discrete-time TGN variant for node classification.

    Per snapshot ``t``:
        1. Select the subgraph induced by nodes with ``time_step <= t``.
        2. Compute a snapshot embedding with a 2-layer GAT.
        3. Update per-node memory with a GRUCell, feeding in the embedding
           concatenated with a time encoding of ``t - last_update[node]``.
        4. Carry the memory forward.

    At the final snapshot, the memory + the latest snapshot embedding are
    concatenated and passed through the classifier head.

    Because the existing ``train_and_evaluate`` helper passes the full
    ``edge_index`` and expects ``model(x, edge_index)``, we expose a
    ``set_time_step`` method and let the caller stash the per-node time
    bucket on the model before training.  This keeps the ``forward``
    signature compatible with the rest of the repo.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int = 128,
        out_channels:    int = 3,
        num_layers:      int = 2,
        heads:           int = 4,
        dropout:         float = 0.4,
        time_dim:        int = 32,
        **kwargs,
    ):
        super().__init__()
        assert hidden_channels % heads == 0

        self.hidden = hidden_channels
        self.dropout = dropout
        head_dim    = hidden_channels // heads

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            GATConv(hidden_channels, head_dim, heads=heads, concat=True,
                    dropout=0.2)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers)
        ])

        self.time_enc = TimeEncoder(time_dim)
        self.mem_proj = nn.Linear(hidden_channels + time_dim, hidden_channels)
        self.mem_cell = nn.GRUCell(hidden_channels, hidden_channels)

        self.classifier = nn.Linear(hidden_channels * 2, out_channels)

        # Caller sets this before training; see ``set_time_step``.
        self._time_step: Optional[torch.Tensor] = None

    def set_time_step(self, time_step: torch.Tensor) -> None:
        """Attach the per-node time bucket tensor before forward()."""
        self._time_step = time_step.long()

    # ---- internal helpers ---------------------------------------------------

    def _snapshot_subgraph(
        self,
        edge_index: torch.Tensor,
        time_step:  torch.Tensor,
        t:          int,
        num_nodes:  int,
    ) -> torch.Tensor:
        """Edge subset with both endpoints having ``time_step <= t``."""
        keep_nodes = time_step <= t
        node_idx   = torch.where(keep_nodes)[0]
        sub_ei, _  = subgraph(
            node_idx, edge_index,
            relabel_nodes=False, num_nodes=num_nodes,
        )
        return sub_ei

    # ---- forward ------------------------------------------------------------

    def forward(self, x, edge_index, edge_attr=None):
        if self._time_step is None:
            raise RuntimeError(
                "SnapshotTGN requires `set_time_step(...)` to be called "
                "with the per-node time bucket tensor before forward()."
            )
        ts      = self._time_step.to(x.device)
        t_min   = int(ts.min().item())
        t_max   = int(ts.max().item())
        n       = x.size(0)

        # Memory vector per node. We rebuild via ``index_copy`` on each
        # update so the previous-step tensor stays on the autograd graph —
        # naive ``memory[touched] = ...`` is in-place and breaks the
        # backward pass on the second epoch.
        memory = x.new_zeros(n, self.hidden)
        # Snapshot at which each node last received a memory update.
        # This one is not on the autograd graph so in-place is fine.
        last_t = torch.full((n,), t_min - 1, dtype=torch.long, device=x.device)

        # Project features once — the encoder is shared across snapshots.
        h_feat = F.relu(self.input_proj(x))
        h_feat = F.dropout(h_feat, p=self.dropout, training=self.training)

        h_snap = h_feat  # will be overwritten each snapshot
        for t in range(t_min, t_max + 1):
            sub_ei = self._snapshot_subgraph(edge_index, ts, t, n)
            h = h_feat
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, sub_ei)
                h = bn(h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h_snap = h

            touched = ts == t
            if touched.any():
                idx = torch.where(touched)[0]
                dt  = (t - last_t[idx]).float()
                t_emb = self.time_enc(dt)
                msg   = torch.cat([h_snap[idx], t_emb], dim=-1)
                msg   = self.mem_proj(msg)
                updated = self.mem_cell(msg, memory[idx])
                memory  = memory.index_copy(0, idx, updated)
                last_t[idx] = t

        return self.classifier(torch.cat([memory, h_snap], dim=-1))

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for c in self.convs: c.reset_parameters()
        for b in self.bns:   b.reset_parameters()
        self.mem_proj.reset_parameters()
        self.mem_cell.reset_parameters()
        self.classifier.reset_parameters()
