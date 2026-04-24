# -*- coding: utf-8 -*-
"""
SkillGraph/graph/graph.py
"""

import os
import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio

from SkillGraph.graph.node import Node
from SkillGraph.agents.agent_registry import AgentRegistry
from SkillGraph.prompt.prompt_set_registry import PromptSetRegistry
from SkillGraph.llm.profile_embedding import get_sentence_embedding
from SkillGraph.gnn.mmgt import MultimodalGraphTransformer, _min_max_norm
from torch_geometric.utils import dense_to_sparse
from SkillGraph.skills.skill_library import SkillLibrary

# ---------------------------------------------------------------------------
# 模块级 CLIP 缓存，避免 copy.deepcopy 时重复加载模型权重
# ---------------------------------------------------------------------------
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_LOAD_ATTEMPTED = False


def _load_clip():
    """懒加载 CLIP。返回 (model, preprocess) 或 (None, None)。"""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_LOAD_ATTEMPTED
    if _CLIP_LOAD_ATTEMPTED:
        return _CLIP_MODEL, _CLIP_PREPROCESS
    _CLIP_LOAD_ATTEMPTED = True
    try:
        import clip
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device="cpu")
        _CLIP_MODEL.eval()
        print("[Graph] CLIP ViT-B/32 loaded for image encoding.")
    except Exception as e:
        print(f"[Graph] CLIP not available ({e}); image_emb will be None (text-only mode).")
        _CLIP_MODEL, _CLIP_PREPROCESS = None, None
    return _CLIP_MODEL, _CLIP_PREPROCESS


class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.
    （接口与原版保持完全兼容，仅内部图预测网络升级为 MMGT。）
    """

    def __init__(
        self,
        domain: str,
        llm_name: Optional[str],
        agent_names: List[str],
        decision_method: str,
        optimized_spatial: bool = False,
        initial_spatial_probability: float = 0.5,
        fixed_spatial_masks: List[List[int]] = None,
        optimized_temporal: bool = False,
        initial_temporal_probability: float = 0.5,
        fixed_temporal_masks: List[List[int]] = None,
        node_kwargs: List[Dict] = None,
        image_feat_dim: int = 768,
        skill_library: Optional[SkillLibrary] = None,  
        constraint_suffix: Optional[str] = None, 
    ):
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [
                [1 if i != j else 0 for j in range(len(agent_names))]
                for i in range(len(agent_names))
            ]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [
                [1 for j in range(len(agent_names))]
                for i in range(len(agent_names))
            ]
        fixed_spatial_masks  = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks) == len(agent_names) ** 2, \
            "fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks) == len(agent_names) ** 2, \
            "fixed_temporal_masks doesn't match the number of agents"

        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.agent_names: List[str] = agent_names
        self.optimized_spatial  = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node: Node = AgentRegistry.get(
            decision_method, **{"domain": self.domain, "llm_name": self.llm_name}
        )
        self.nodes: Dict[str, Node] = {}
        self.potential_spatial_edges:  List[List[str]] = []
        self.potential_temporal_edges: List[List[str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]

        self.skill_library = skill_library
        self.constraint_suffix = constraint_suffix

        self.init_nodes()
        self.init_potential_edges()

        self.prompt_set      = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features        = self.construct_features()
        self.image_feat_dim  = image_feat_dim

        self.mmgt = MultimodalGraphTransformer(
            node_feat_dim  = self.features.size(1),  # sentence-bert = 384
            text_query_dim = self.features.size(1),
            image_feat_dim = image_feat_dim,          # CLIP = 768
            d_model        = 128,
            num_gt_layers  = 2,
            num_heads      = 4,
            dropout        = 0.1,
        )

        # spatial logits 占位（arun 中每次前向传播都会更新）
        self.spatial_logits: torch.Tensor = torch.zeros(
            len(self.potential_spatial_edges)
        )

        init_spatial_logit = (
            torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability)))
            if optimized_spatial else 10.0
        )
        self.spatial_masks = torch.nn.Parameter(
            fixed_spatial_masks, requires_grad=False
        )

        init_temporal_logit = (
            torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability)))
            if optimized_temporal else 10.0
        )
        self.temporal_logits = torch.nn.Parameter(
            torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal)
            * init_temporal_logit,
            requires_grad=optimized_temporal,
        )
        self.temporal_masks = torch.nn.Parameter(
            fixed_temporal_masks, requires_grad=False
        )

    # ------------------------------------------------------------------
    # 图像编码（懒加载 CLIP，无 CLIP 时返回 None）
    # ------------------------------------------------------------------

    def _encode_image(self, image_path):
        if not image_path or not os.path.exists(str(image_path)):
            return None
        clip_model, clip_preprocess = _load_clip()
        if clip_model is None:
            return None
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path).convert("RGB")
            img_tensor = clip_preprocess(img).unsqueeze(0)  # [1, 3, H, W]

            with torch.no_grad():
                visual = clip_model.visual
                x = visual.conv1(img_tensor.to(next(visual.parameters()).dtype))
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [1, P, D]
                x = torch.cat([
                    visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1),
                    x
                ], dim=1)
                x = x + visual.positional_embedding
                x = visual.ln_pre(x)
                x = x.permute(1, 0, 2)   # NLD -> LND
                x = visual.transformer(x)
                x = x.permute(1, 0, 2)   # LND -> NLD

                patch_tokens = x[0, 1:, :].float()  # [49, 768]
                return patch_tokens

        except Exception as e:
            print(f"[Graph] Patch token extraction failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Graph 构建辅助
    # ------------------------------------------------------------------

    def construct_adj_matrix(self):
        role_connect: List[Tuple[str, str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes, num_nodes))
        role_2_id: Dict[str, List[int]] = {}

        for in_role, out_role in role_connect:
            role_2_id.setdefault(in_role, [])
            role_2_id.setdefault(out_role, [])
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            if role in role_2_id:
                role_2_id[role].append(i)

        for in_role, out_role in role_connect:
            for in_id in role_2_id.get(in_role, []):
                for out_id in role_2_id.get(out_role, []):
                    role_adj[in_id][out_id] = 1

        edge_index, _ = dense_to_sparse(role_adj)
        return edge_index

    def construct_features(self):
        """
        ★ 修复：优先用 agent 当前 skill 的 description 作为节点特征文本。
        skill 切换后（_select_skill_for_query），特征向量能反映最新角色语义。
        无 skill 时退化为原来的 prompt_set.get_description(role)。
        """
        features = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if (hasattr(node, 'current_skill') and node.current_skill is not None):
                profile = node.current_skill.to_retrieval_text()
            else:
                profile = self.prompt_set.get_description(node.role)
            features.append(get_sentence_embedding(profile))
        return torch.tensor(np.array(features))


    def construct_new_features(self, query):
        """向后兼容保留；MMGT 路径中不再使用。"""
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes, 1))
        return torch.cat((self.features, query_embedding), dim=1)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors:
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        return sum(len(node.spatial_successors) for node in self.nodes.values())

    @property
    def num_nodes(self):
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Node 管理
    # ------------------------------------------------------------------

    def find_node(self, id: str):
        if id in self.nodes:
            return self.nodes[id]
        raise Exception(
            f"Node not found: {id} among {[n.id for n in self.nodes.values()]}"
        )

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"]        = self.domain
                kwargs["llm_name"]      = self.llm_name
                kwargs["skill_library"] = self.skill_library  
                kwargs["constraint_suffix"] = self.constraint_suffix
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def init_potential_edges(self):
        for node1_id in self.nodes:
            for node2_id in self.nodes:
                self.potential_spatial_edges.append([node1_id, node2_id])
                self.potential_temporal_edges.append([node1_id, node2_id])

    # ------------------------------------------------------------------
    # Connection 管理
    # ------------------------------------------------------------------

    def clear_spatial_connection(self):
        for node_id in self.nodes:
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors   = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors   = []

    def clear_temporal_connection(self):
        for node_id in self.nodes:
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors   = []

    def connect_decision_node(self):
        for node_id in self.nodes:
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None):
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]

        for potential_connection, edge_logit, edge_mask in zip(
            self.potential_spatial_edges, self.spatial_logits, self.spatial_masks
        ):
            out_node: Node = self.find_node(potential_connection[0])
            in_node:  Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and not self.optimized_spatial:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node, 'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        return torch.sum(torch.stack(log_probs))

    def construct_temporal_connection(self, round: int = 0, temperature: float = 1.0, threshold: float = None):
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))
        for potential_connection, edge_logit, edge_mask in zip(
            self.potential_temporal_edges, self.temporal_logits, self.temporal_masks
        ):
            out_node: Node = self.find_node(potential_connection[0])
            in_node:  Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and not self.optimized_temporal:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node, 'temporal')
                continue
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node, 'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))

        return torch.sum(torch.stack(log_probs))

    # ------------------------------------------------------------------
    # 同步 run（保持不变，供非 VL 任务使用）
    # ------------------------------------------------------------------

    def run(self, inputs: Any, num_rounds: int = 3, max_tries: int = 3, max_time: int = 600):
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)

            in_degree = {
                node_id: len(node.spatial_predecessors)
                for node_id, node in self.nodes.items()
            }
            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]
            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs)
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes:
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            self.update_memory()

        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if not final_answers:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs

    # ------------------------------------------------------------------
    # 异步 arun（核心改动：GCN → MMGT，支持图文输入）
    # ------------------------------------------------------------------

    async def arun(
        self,
        input: Dict[str, Any],
        num_rounds: int = 3,
        max_tries: int = 3,
        max_time: int = 600,
    ) -> Tuple[List[Any], torch.Tensor]:
        """
        input 格式：
            {'task': str, 'image': str | None}
            'image' 字段为本地图片路径；不存在或为空时退化为纯文本。
        """
        log_probs = 0

        # ★ 修复：先让每个 agent 根据当前 query 检索并切换技能，
        #         再重建特征，确保 MMGT 看到的 Xagent 与本轮技能一致
        #         （对应论文 Algorithm 1 第 7 行：先 Retrieve skill，再 rebuild Xagent）

        query_text    = input['task']
        analyze_nodes = [n for n in self.nodes.values() if hasattr(n, '_select_skill_for_query')]

        # ★ 统一检索一次，按 agent 序号依次分配不同排名的技能
        shared_matches = None
        if analyze_nodes and getattr(analyze_nodes[0], 'skill_library', None) is not None:
            shared_matches = analyze_nodes[0].skill_library.get_skills_by_query(
                query_text, top_k=len(analyze_nodes)
            )

        for rank, node in enumerate(analyze_nodes):
            node._select_skill_for_query(
                query_text,
                shared_matches=shared_matches,
                agent_rank=rank,
            )

        self.features = self.construct_features()

        # ---- MMGT 前向传播：预测 spatial edge logits ----
        query_text_emb  = torch.tensor(get_sentence_embedding(query_text))
        query_image_emb = self._encode_image(input.get('image'))

        self.spatial_logits = self.mmgt(
            node_features   = self.features,
            role_adj_matrix = self.role_adj_matrix,
            query_text_emb  = query_text_emb,
            query_image_emb = query_image_emb,
        )  # [N*N]，归一化到 [-1, 1]

        # ---- 原有的图执行逻辑（完全不变）----
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)

            in_degree = {
                node_id: len(node.spatial_predecessors)
                for node_id, node in self.nodes.items()
            }
            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]
            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(
                            self.nodes[current_node_id].async_execute(input),
                            timeout=max_time,
                        )
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes:
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            self.update_memory()

        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if not final_answers:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs


    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def update_memory(self):
        for node in self.nodes.values():
            node.update_memory()

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        return any(self.check_cycle(s, target_nodes) for s in new_node.spatial_successors)

    def update_masks(self, pruning_rate: float):
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num = max(1, int(torch.round(num_edges * pruning_rate).item()))
            _logits = self.spatial_logits.clone()
            _logits[self.spatial_masks == 0] = _logits.min() - 1.0
            prune_idx = torch.argsort(_logits)[:prune_num + int(num_masks.item())]
            self.spatial_masks[prune_idx] = 0

        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num = max(1, int(torch.round(num_edges * pruning_rate).item()))
            _logits = self.temporal_logits.clone()
            _logits[self.temporal_masks == 0] = _logits.min() - 1.0
            prune_idx = torch.argsort(_logits)[:prune_num + int(num_masks.item())]
            self.temporal_masks[prune_idx] = 0

        return self.spatial_masks, self.temporal_masks


# ---------------------------------------------------------------------------
# 向后兼容：保留模块级 min_max_norm（其他模块可能 import 它）
# ---------------------------------------------------------------------------

def min_max_norm(tensor: torch.Tensor) -> torch.Tensor:
    return _min_max_norm(tensor)
