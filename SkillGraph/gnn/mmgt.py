# -*- coding: utf-8 -*-
"""
SkillGraph/gnn/mmgt.py

Multimodal Graph Transformer (MMGT)

Architecture
------------
1. MultimodalQueryEncoder   Encodes text + image into the initial embedding of a virtual node
2. PerAgentImageAttention   Each agent uses its own role embedding as the query to independently cross-attend to image patches
3. GraphTransformerLayer    Performs message passing with attention, using role_adj as a structural bias prior
4. VirtualNodeLayer         Enables bidirectional cross-attention between the virtual node and all real nodes (as intended in the paper)
5. MultimodalGraphTransformer  The sole external interface; produces outputs fully compatible with the original spatial_logits format
------------------------------------------------
    self.spatial_logits = self.mmgt(
        node_features    = self.features,          # [N, node_feat_dim]
        role_adj_matrix  = self.role_adj_matrix,   # [2, E] edge_index (sparse)
        query_text_emb   = text_emb,               # [text_query_dim]
        query_image_emb  = image_emb,              # [P, image_feat_dim] | [image_feat_dim] | None
    )                                              # Returns [N*N], normalized to [-1, 1]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Multimodal Query Encoder
#    Uses text as the query and image patches as the key/value for fusion via cross-attention
#    Automatically falls back to a text-only pathway when no image is provided, ensuring backward compatibility
# ---------------------------------------------------------------------------

class MultimodalQueryEncoder(nn.Module):
    def __init__(
        self,
        text_dim: int = 384,
        image_dim: int = 512,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.text_proj  = nn.Linear(text_dim, d_model)
        self.image_proj = nn.Linear(image_dim, d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.text_only_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.norm    = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        text_emb: torch.Tensor,                    # [text_dim] or [1, text_dim]
        image_emb: Optional[torch.Tensor] = None,  # [img_dim] or [P, img_dim] or None
    ) -> torch.Tensor:
        """返回 [d_model]"""
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)   # [1, text_dim]

        if image_emb is None:
            out = self.text_only_proj(text_emb)
            return self.norm(out).squeeze(0)   # [d_model]

        if image_emb.dim() == 1:
            image_emb = image_emb.unsqueeze(0)    # [1, img_dim]

        text_q   = self.text_proj(text_emb)        # [1, d_model]
        image_kv = self.image_proj(image_emb)      # [P, d_model]

        text_q   = text_q.unsqueeze(0)             # [1, 1, d_model]
        image_kv = image_kv.unsqueeze(0)           # [1, P, d_model]

        fused, _ = self.cross_attn(text_q, image_kv, image_kv)  # [1, 1, d_model]
        fused = fused.squeeze(0)                   # [1, d_model]

        fused = fused + self.text_proj(text_emb)
        fused = self.norm(fused)
        return self.out_proj(fused).squeeze(0)     # [d_model]


# ---------------------------------------------------------------------------
# 2. Per-Agent Image Attention
#    Each agent uses its own role embedding as the query and independently performs cross-attention over image patches
#    → Different agents naturally attend to different regions of the image (e.g., OCR, counting, color, etc.)
# ---------------------------------------------------------------------------

class PerAgentImageAttention(nn.Module):

    def __init__(self, d_model=128, image_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, d_model)
        self.role_proj  = nn.Linear(d_model, d_model)
        
        self.vn_gate    = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        node_feats: torch.Tensor,                  # [N, d_model]
        image_patches: Optional[torch.Tensor],     # [P, image_dim] or None
        virtual_node: Optional[torch.Tensor] = None,  # [d_model]
    ) -> torch.Tensor:
        if image_patches is None:
            return node_feats

        queries = self.role_proj(node_feats)        # [N, d_model]
        
        # use the virtual node to modulate each agent's query. The virtual node carries the global task-level focus of attention, and the gate determines how much each agent should borrow from it
        if virtual_node is not None:
            vn_expanded = virtual_node.unsqueeze(0).expand_as(queries)  # [N, d_model]
            gate = self.vn_gate(torch.cat([queries, vn_expanded], dim=-1))  # [N, d_model]
            queries = queries + gate * vn_expanded   # Soft fusion while preserving role-specific characteristics
        
        keys = self.image_proj(image_patches)       # [P, d_model]
        q  = queries.unsqueeze(0)
        kv = keys.unsqueeze(0)
        out, _ = self.cross_attn(q, kv, kv)
        out = out.squeeze(0)
        return self.norm(node_feats + out)

# ---------------------------------------------------------------------------
# 3. Graph Transformer Layer
#    Uses attention weights for message passing; role_adj serves as a structural prior bias
#    (a soft constraint rather than a hard cutoff)
# ---------------------------------------------------------------------------

class GraphTransformerLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                           # [N, d_model]
        attn_bias: Optional[torch.Tensor] = None,  # [N, N]
    ) -> torch.Tensor:
        x_b = x.unsqueeze(0)                       # [1, N, d_model]
        attn_mask = attn_bias if attn_bias is not None else None

        out, _ = self.attn(x_b, x_b, x_b, attn_mask=attn_mask)
        out = out.squeeze(0)
        x   = self.norm1(x + out)
        x   = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# 4. Virtual Node Layer
#    implements the bidirectional message passing of the task-specific virtual node:
#      ① All real nodes → virtual node   (the virtual node gathers the collective state of the agents)
#      ② virtual node → all real nodes   (broadcasts task-specific conditions)
# ---------------------------------------------------------------------------


class VirtualNodeLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.nodes_to_vn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.vn_to_nodes = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_v  = nn.LayerNorm(d_model)
        self.norm_n  = nn.LayerNorm(d_model)
        self.ffn_v   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model),
        )
        self.norm_v2 = nn.LayerNorm(d_model)

    def forward(
        self,
        node_feats: torch.Tensor,    # [N, d_model]
        virtual_node: torch.Tensor,  # [d_model]
    ):
        n = node_feats.unsqueeze(0)                      # [1, N, d_model]
        v = virtual_node.unsqueeze(0).unsqueeze(0)       # [1, 1, d_model]

        # All real nodes → virtual node 
        v_new, _ = self.nodes_to_vn(v, n, n)             # [1, 1, d_model]
        virtual_node = self.norm_v(virtual_node + v_new.squeeze())
        virtual_node = self.norm_v2(virtual_node + self.ffn_v(virtual_node))

        # virtual node → all real nodes
        v_b = virtual_node.unsqueeze(0).unsqueeze(0)     # [1, 1, d_model]
        n_new, _ = self.vn_to_nodes(n, v_b, v_b)        # [1, N, d_model]
        node_feats = self.norm_n(node_feats + n_new.squeeze(0))

        return node_feats, virtual_node                  # [N, d_model], [d_model]


class MultimodalGraphTransformer(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 384,
        text_query_dim: int = 384,
        image_feat_dim: int = 512,
        d_model: int = 128,
        num_gt_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model       = d_model
        self.image_feat_dim = image_feat_dim

        self.query_encoder = MultimodalQueryEncoder(
            text_dim=text_query_dim,
            image_dim=image_feat_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.image_attn = PerAgentImageAttention(
            d_model=d_model,
            image_dim=image_feat_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.gt_layers = nn.ModuleList([
            GraphTransformerLayer(d_model, num_heads, dropout)
            for _ in range(num_gt_layers)
        ])
        self.vn_layers = nn.ModuleList([
            VirtualNodeLayer(d_model, num_heads, dropout)
            for _ in range(num_gt_layers)
        ])

        self.edge_predictor = nn.Bilinear(d_model, d_model, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight)

    def _build_attn_bias(
        self,
        edge_index: torch.Tensor,   # [2, E]
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts a sparse edge_index into a dense attention bias matrix of shape [N, N].
        Positions with edges in role_adj receive a bias of 0 (no interference with attention),
        while positions without edges receive a bias of -1e4 (soft suppression, still allowing cross-role learning).
        """
        bias = torch.full((N, N), -1e4, device=device)
        bias.fill_diagonal_(0.0)
        if edge_index.numel() > 0:
            bias[edge_index[0], edge_index[1]] = 0.0
        return bias

    def forward(self, node_features, role_adj_matrix, query_text_emb, query_image_emb=None):
        N      = node_features.size(0)
        device = node_features.device

        virtual_node = self.query_encoder(query_text_emb, query_image_emb)

        h = self.node_proj(node_features)

        # Step 3: Per-agent image attention, now with virtual_node passed in
        #         After knowing where the overall task is focusing, each agent decides where to attend on its own
        h = self.image_attn(h, query_image_emb, virtual_node=virtual_node)  

        attn_bias = self._build_attn_bias(role_adj_matrix, N, device)
        for gt_layer, vn_layer in zip(self.gt_layers, self.vn_layers):
            h = gt_layer(h, attn_bias)
            h, virtual_node = vn_layer(h, virtual_node)

        h_i = h.unsqueeze(1).expand(N, N, -1).contiguous().reshape(N * N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1).contiguous().reshape(N * N, -1)
        logits = self.edge_predictor(h_i, h_j).squeeze(-1)
        return _min_max_norm(logits)



def _min_max_norm(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min()
    max_val = tensor.max()
    if (max_val - min_val).abs() < 1e-8:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val) * 2 - 1
