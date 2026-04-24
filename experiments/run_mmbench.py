# -*- coding: utf-8 -*-
"""
Filename: run_mmbench.py

Quick start
-----------
export HF_ENDPOINT=https://hf-mirror.com 
python experiments/run_mmbench.py --mode FullConnected --batch_size 4 --agent_nums 4 --num_iterations 5 --optimized_spatial --evolve_skills

python experiments/run_mmbench.py \\
    --mode FullConnected --batch_size 4 --agent_nums 4 \\
    --num_iterations 10 --evolve_skills


python experiments/run_mmbench.py --mode DirectAnswer --agent_nums 1
python experiments/run_mmbench.py --mode Chain --agent_nums 4
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.utils.data import random_split
import asyncio
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from SkillGraph.graph.graph import Graph
from skillgraph_datasets.mmbench_dataset import MMBenchDataset
from experiments.train_vl import train
from experiments.evaluate_vl import evaluate
from SkillGraph.utils.const import SkillGraph_ROOT
from SkillGraph.skills.skill_library import SkillLibrary
from SkillGraph.skills.skill_designer import SkillDesigner
from SkillGraph.llm.llm_registry import LLMRegistry
from SkillGraph.skills.skill_library import SkillLibrary, attach_tools_to_library


class _DatasetSubset(torch.utils.data.Dataset):
    """
    Wraps a torch Subset and delegates MMBenchDataset-specific methods
    (record_to_input, record_to_target_answer, postprocess_answer)
    to the underlying base dataset.
    """
    def __init__(self, subset: torch.utils.data.Subset):
        self._subset = subset
        self._base   = subset.dataset  # 原始 MMBenchDataset

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        return self._subset[idx]

    def __getattr__(self, name):
        return getattr(self._base, name)

def parse_args():
    parser = argparse.ArgumentParser(description='SkillGraph (MMGT) on MMBench benchmark')

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=[
                            'DirectAnswer', 'FullConnected', 'Random', 'Chain',
                            'Debate', 'Layered', 'Star', 'Mesh',
                            'FakeFullConnected', 'FakeRandom', 'FakeChain',
                            'FakeStar', 'FakeMesh', 'FakeAGRandom', 'FakeAGFull',
                        ],
                        help='Graph topology mode. Default: FullConnected')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='Training batch size')
    parser.add_argument('--agent_names', nargs='+', type=str,
                        default=['AnalyzeAgent'],
                        help='Agent type names')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Number of agents for each name in --agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of optimisation iterations (default 10)')
    parser.add_argument('--imp_per_iterations', type=int, default=1,
                        help='Prune / evolve every N iterations (default 5)')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Reasoning rounds per query (default 1)')
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help='Edge pruning rate (default 0.25)')
    parser.add_argument('--llm_name', type=str, default='gpt-4o',
                        help='Model name (default gpt-4o)')
    parser.add_argument('--domain', type=str, default='mmbench',
                        help='Domain / dataset name (default mmbench)')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='Final node decision method (default FinalRefer)')

    # ---- 训练开关 ----
    parser.add_argument('--optimized_spatial',  action='store_true',
                        help='Enable spatial topology learning via MMGT.')
    parser.add_argument('--optimized_temporal', action='store_true',
                        help='Enable temporal topology learning.')
    parser.add_argument('--evolve_skills', action='store_true',
                        help=(
                            'Enable skill evolution via SkillDesigner. '
                            'Can be used independently of --optimized_spatial / '
                            '--optimized_temporal.'
                        ))

    # ---- 数据集相关 ----
    parser.add_argument('--train_jsonl', type=str, default=None,
                        help='Explicit data.jsonl for training split.')
    parser.add_argument('--val_jsonl', type=str, default=None,
                        help='Explicit data.jsonl for validation split.')
    parser.add_argument('--out_root', type=str, default='./datasets_out',
                        help='Root dir where prepared data is stored (default ./datasets_out)')
    parser.add_argument('--dataset_id', type=str,
                        default='HuggingFaceM4/MMBench_dev',
                        help='HuggingFace dataset id')
    parser.add_argument('--train_split', type=str, default='train',
                        help='HuggingFace split used for training (default train)')
    parser.add_argument('--val_split', type=str, default='train',
                        help='HuggingFace split used for validation (default train)')
    parser.add_argument('--limit_train', type=int, default=None,
                        help='Max training examples (default: all)')
    parser.add_argument('--hf_endpoint', type=str, default=None,
                        help='HuggingFace mirror, e.g. https://hf-mirror.com')
    parser.add_argument('--limit_questions', type=int, default=None,
                        help='Max validation questions to evaluate (default: all)')

    parser.add_argument(
        '--image_feat_dim', type=int, default=768,
        help=(
            'Image feature dimension passed to MMGT. '
            'Default 768 = CLIP ViT-B/16 patch tokens. '
            'Use 512 for ViT-B/32 global vector.'
        ),
    )

    args = parser.parse_args()

    if len(args.agent_names) != len(args.agent_nums):
        parser.error('--agent_names and --agent_nums must have the same length.')

    result_path = SkillGraph_ROOT / 'result'
    os.makedirs(result_path, exist_ok=True)
    return args


def get_kwargs(mode: str, N: int) -> dict:
    initial_spatial_probability: float  = 0.5
    fixed_spatial_masks                 = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks                = None
    node_kwargs                         = None

    def generate_layered_graph(N, layer_num=2):
        adj = [[0] * N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj[i][j] = 1
        return adj

    def generate_mesh_graph(N):
        adj = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                adj[i][j] = 1
        return adj

    def generate_star_graph(N):
        adj = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj[0][i] = 1
        return adj

    if mode == 'DirectAnswer':
        fixed_spatial_masks  = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Normal'}]
    elif mode in ('FullConnected', 'FakeFullConnected', 'FakeAGFull'):
        fixed_spatial_masks  = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1] * N for _ in range(N)]
    elif mode in ('Random', 'FakeRandom', 'FakeAGRandom'):
        fixed_spatial_masks  = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode in ('Chain', 'FakeChain'):
        fixed_spatial_masks  = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks  = [[0] * N for _ in range(N)]
        fixed_temporal_masks = [[1] * N for _ in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks  = generate_layered_graph(N)
        fixed_temporal_masks = [[1] * N for _ in range(N)]
    elif mode in ('Mesh', 'FakeMesh'):
        fixed_spatial_masks  = generate_mesh_graph(N)
        fixed_temporal_masks = [[1] * N for _ in range(N)]
    elif mode in ('Star', 'FakeStar'):
        fixed_spatial_masks  = generate_star_graph(N)
        fixed_temporal_masks = [[1] * N for _ in range(N)]

    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [
            {'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Normal'}
            for i in range(N)
        ]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [
            {'role': 'Fake'} if i % 2 == N % 2 else {'role': None}
            for i in range(N)
        ]

    return {
        'initial_spatial_probability':  initial_spatial_probability,
        'fixed_spatial_masks':          fixed_spatial_masks,
        'initial_temporal_probability': initial_temporal_probability,
        'fixed_temporal_masks':         fixed_temporal_masks,
        'node_kwargs':                  node_kwargs,
    }

def _save_results(
    results: list,
    score: float,
    args,
    timestamp: str,
    train_log_path: Optional[str] = None,  # 关联训练记录
) -> str:
    result_path = SkillGraph_ROOT / 'result'
    llm_tag  = args.llm_name.replace('/', '_').replace(':', '_')
    out_file = result_path / f'eval_{args.mode}_{llm_tag}_{timestamp}.jsonl'

    with open(out_file, 'w', encoding='utf-8') as f:
        summary = {
            "_type":            "summary",
            "score":            round(score, 6),
            "correct":          sum(1 for r in results if r["is_correct"]),
            "total":            len(results),
            "mode":             args.mode,
            "llm_name":         args.llm_name,
            "num_rounds":       args.num_rounds,
            "image_feat_dim":   args.image_feat_dim,
            "optimized_spatial":  args.optimized_spatial,
            "optimized_temporal": args.optimized_temporal,
            "evolve_skills":    args.evolve_skills,
            "timestamp":        timestamp,
            "train_log":        train_log_path,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        for r in results:
            f.write(json.dumps({"_type": "sample", **r}, ensure_ascii=False) + '\n')

    return str(out_file)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    agent_names = [
        name
        for name, num in zip(args.agent_names, args.agent_nums)
        for _ in range(num)
    ]
    kwargs = get_kwargs(args.mode, len(agent_names))

    skill_library  = SkillLibrary(domain=args.domain)
    attach_tools_to_library(skill_library)    

    llm_instance   = LLMRegistry.get(args.llm_name)
    skill_designer = SkillDesigner(skill_library=skill_library, llm=llm_instance)

    common_kwargs = dict(
        out_root=args.out_root,
        dataset_id=args.dataset_id,
        hf_endpoint=args.hf_endpoint,
    )
    dataset_all = MMBenchDataset(
        split='train',
        data_jsonl=args.train_jsonl,
        limit=args.limit_train,
        **common_kwargs,
    )
    total      = len(dataset_all)
    train_size = int(total * 0.8)
    val_size   = total - train_size

    constraint = dataset_all.get_constraint_suffix()

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        image_feat_dim=args.image_feat_dim,
        skill_library=skill_library,
        constraint_suffix=constraint, 
        **kwargs,
    )

    g = torch.Generator().manual_seed(42)
    _train_subset, _val_subset = random_split(
        dataset_all, [train_size, val_size], generator=g
    )
    dataset_train = _DatasetSubset(_train_subset)
    dataset_val   = _DatasetSubset(_val_subset)

    should_train = (
        args.optimized_spatial
        or args.evolve_skills
    )

    train_log_path: Optional[str] = None

    if should_train:
        llm_tag        = args.llm_name.replace('/', '_').replace(':', '_')
        train_log_path = str(
            SkillGraph_ROOT / 'result' / f'train_{args.mode}_{llm_tag}_{timestamp}.jsonl'
        )
        print(
            f'[run_mmbench] Training: '
            f'optimized_spatial={args.optimized_spatial}, '
            f'optimized_temporal={args.optimized_temporal}, '
            f'evolve_skills={args.evolve_skills}'
        )
        train_log_path = await train(
            graph=graph,
            dataset=dataset_train,
            num_iters=args.num_iterations,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=args.batch_size,
            skill_library=skill_library,
            skill_designer=skill_designer if args.evolve_skills else None,
            evolve_every=args.imp_per_iterations,
            train_log_path=train_log_path,
        )

    # ---- 评估 ----
    score, results = await evaluate(
        graph=graph,
        dataset=dataset_val,
        num_rounds=args.num_rounds,
        limit_questions=args.limit_questions,
        eval_batch_size=args.batch_size,
        skill_library=skill_library,
    )

    out_file = _save_results(results, score, args, timestamp, train_log_path)
    print(f'Score: {score:.4f}')
    print(f'Eval results saved : {out_file}')
    if train_log_path:
        print(f'Train log saved    : {train_log_path}')


if __name__ == '__main__':
    asyncio.run(main())

