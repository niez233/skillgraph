# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch

from SkillGraph.graph.graph import Graph
from experiments.accuracy import Accuracy
from SkillGraph.utils.globals import Cost, PromptTokens, CompletionTokens


async def train(
    graph: Graph,
    dataset,
    num_iters: int = 100,
    num_rounds: int = 1,
    lr: float = 0.1,
    batch_size: int = 4,
    skill_library=None,
    skill_designer=None,
    evolve_every: int = 5,
    train_log_path: Optional[str] = None,
) -> Optional[str]:
    # ── Init log file ──────────────────────────────────────────────────────
    log_path: Optional[Path] = None
    if train_log_path:
        log_path = Path(train_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "_type":        "train_start",
                "num_iters":    num_iters,
                "batch_size":   batch_size,
                "lr":           lr,
                "evolve_every": evolve_every,
            }, ensure_ascii=False) + '\n')

    def _append_log(record: dict) -> None:
        if log_path is None:
            return
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # ── Data loader ────────────────────────────────────────────────────────
    def infinite_data_loader() -> Iterator:
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                yield dataset[idx.item()]

    loader    = infinite_data_loader()
    optimizer = torch.optim.Adam(graph.mmgt.parameters(), lr=lr)
    graph.mmgt.train()

    overall_accuracy = Accuracy()

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        start_ts = time.time()

        correct_answers:  List[str]   = []
        answer_log_probs: List        = []
        records_in_batch: List        = []
        realized_graphs:  List[Graph] = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph      = copy.deepcopy(graph)
            realized_graph.mmgt = graph.mmgt

            if skill_library is not None:
                realized_graph.skill_library = skill_library
                for agent in realized_graph.nodes.values():
                    if hasattr(agent, "skill_library"):
                        agent.skill_library = skill_library

            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(
                asyncio.create_task(realized_graph.arun(input_dict, num_rounds))
            )
            correct_answers.append(dataset.record_to_target_answer(record))
            records_in_batch.append(record)
            realized_graphs.append(realized_graph)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)

        loss_list:   List[torch.Tensor] = []
        utilities:   List[float]        = []
        answers:     List[str]          = []
        sample_logs: List[dict]         = []
        record_skill_tasks: List        = []

        for idx, (raw_answer, log_prob, correct_answer, record, rg) in enumerate(
            zip(raw_answers, log_probs, correct_answers, records_in_batch, realized_graphs)
        ):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)

            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            overall_accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)

            loss_list.append(-log_prob * utility)

            skill_names = {
                agent_id: (
                    agent.current_skill.skill_name
                    if getattr(agent, 'current_skill', None) else None
                )
                for agent_id, agent in rg.nodes.items()
            }

            # ★ 用 agent_id 作为 key，与下面的取值循环保持一致
            agent_answers = {
                agent_id: (
                    agent.outputs[-1]
                    if isinstance(agent.outputs, list) and agent.outputs
                    else (agent.outputs or "")
                )
                for agent_id, agent in rg.nodes.items()
            }
            agent_answers["__decision__"] = (
                rg.decision_node.outputs[-1]
                if isinstance(rg.decision_node.outputs, list) and rg.decision_node.outputs
                else (rg.decision_node.outputs or "")
            )

            sample_logs.append({
                "id":            str(record.get("id", f"iter{i_iter}_idx{idx}")),
                "question":      str(record.get("question", "")),
                "predicted":     answer,
                "correct":       correct_answer,
                "is_correct":    utility > 0,
                "skill_names":   skill_names,
                "agent_answers": agent_answers,
            })

            if skill_library is not None:
                question_id = str(record.get("index", f"iter{i_iter}_idx{idx}"))
                # ★ 从 record 里取 choices 和 image_path
                choices    = record.get("choices", {}) or {}
                image_path = str(record.get("image", "") or "")

                # ★ 用 agent_id 直接作为 key 取完整推理文本，避免 agent.id 不一致的问题
                for agent_id, agent in rg.nodes.items():
                    if hasattr(agent, "record_skill_result"):
                        agent_full_response = agent_answers.get(agent_id, "")
                        record_skill_tasks.append(
                            asyncio.create_task(
                                agent.record_skill_result(
                                    is_correct      = (utility > 0),
                                    question_id     = question_id,
                                    question_text   = str(record.get("question", "")),
                                    choices         = choices,        # ★ 新增
                                    image_path      = image_path,     # ★ 新增
                                    model_answer    = answer,
                                    gold_answer     = correct_answer,
                                    model_reasoning = agent_full_response,  # ★ 现在能正确取到
                                )
                            )
                        )

        # 等待所有 lesson 生成完成，再做反向传播
        if record_skill_tasks:
            await asyncio.gather(*record_skill_tasks)

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        elapsed    = time.time() - start_ts
        did_evolve = bool(skill_designer and (i_iter + 1) % evolve_every == 0)

        print("raw_answers:", raw_answers)
        print("answers:", answers)
        print(f"Batch time {elapsed:.3f}s  utilities: {utilities}  loss: {total_loss.item():.4f}")

        _append_log({
            "_type":             "iter",
            "iter":              i_iter,
            "loss":              round(total_loss.item(), 6),
            "batch_accuracy":    round(float(np.mean([u > 0 for u in utilities])), 4),
            "running_accuracy":  round(overall_accuracy.get(), 4),
            "utilities":         [round(u, 4) for u in utilities],
            "elapsed_sec":       round(elapsed, 3),
            "skill_evolved":     did_evolve,
            "prompt_tokens":     PromptTokens.instance().value,
            "completion_tokens": CompletionTokens.instance().value,
            "cost":              Cost.instance().value,
            "samples":           sample_logs,
        })

        if did_evolve:
            print(f"[Train] Iter {i_iter + 1}: triggering skill evolution...")
            await skill_designer.evolve()

    # ── Training summary ───────────────────────────────────────────────────
    _append_log({
        "_type":          "train_summary",
        "total_iters":    num_iters,
        "final_accuracy": round(overall_accuracy.get(), 4),
        "final_cost":     Cost.instance().value,
    })

    print(f"[Train] Log written to: {log_path}")
    return str(log_path) if log_path else None
