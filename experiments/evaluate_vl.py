# -*- coding: utf-8 -*-
"""
Filename: evaluate_vl.py

Evaluation loop for vision-language datasets (MME, MMBench).
"""

from __future__ import annotations

import asyncio
import copy
import math
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from SkillGraph.graph.graph import Graph
from experiments.accuracy import Accuracy
from SkillGraph.utils.globals import Cost, PromptTokens, CompletionTokens
from SkillGraph.skills.skill_library import SkillLibrary


async def evaluate(
    graph: Graph,
    dataset,
    num_rounds: int = 1,
    limit_questions: Optional[int] = None,
    eval_batch_size: int = 4,
    skill_library: Optional[SkillLibrary] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate graph on a vision-language dataset.

    Returns
    -------
    (score, results)
        score   : float, accuracy in [0, 1]
        results : per-sample list, each containing
                  id / question / predicted / correct / is_correct / reason / raw_answer
    """
    print(f"Evaluating SkillGraph on {dataset.__class__.__name__} split={dataset.split}")

    graph.mmgt.eval()

    accuracy = Accuracy()
    results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Batch loader
    # ------------------------------------------------------------------ #
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records: List[Any] = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = (
        min(len(dataset), limit_questions)
        if limit_questions is not None
        else len(dataset)
    )
    num_batches = int(math.ceil(data_len / eval_batch_size))

    # ------------------------------------------------------------------ #
    # Main evaluation loop
    # ------------------------------------------------------------------ #
    for i_batch, record_batch in tqdm(
        enumerate(eval_loader(batch_size=eval_batch_size)),
        total=num_batches,
    ):
        print(80 * "-")
        start_ts = time.time()
        answer_log_probs  = []
        realized_graphs: List[Graph] = []

        for record in record_batch:
            realized_graph      = copy.deepcopy(graph)
            realized_graph.mmgt = graph.mmgt  # share weights

            if skill_library is not None:
                realized_graph.skill_library = skill_library
                for agent in realized_graph.nodes.values():
                    if hasattr(agent, 'skill_library'):
                        agent.skill_library = skill_library

            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(
                asyncio.create_task(realized_graph.arun(input_dict, num_rounds))
            )
            realized_graphs.append(realized_graph)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)

        print(f"Batch time {time.time() - start_ts:.3f}")

        for raw_answer, record, rg in zip(raw_answers, record_batch, realized_graphs):
            print("Raw answer:", raw_answer)
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            print("Correct answer:", correct_answer)

            raw_text = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else ""
            reason   = raw_text.strip()[1:].lstrip(". ：:").strip()
            print("Reason:", reason)

            is_correct = (answer == correct_answer)
            accuracy.update(answer, correct_answer)
            accuracy.print()

            # ★ Pass model_answer + gold_answer so Designer can see what went wrong
            if skill_library is not None:
                question_id = str(record.get("index", record.get("id", "")))
                for agent in rg.nodes.values():
                    if hasattr(agent, "record_skill_result"):
                        agent.record_skill_result(
                            is_correct=   is_correct,
                            question_id=  question_id,
                            question_text=str(record.get("question", "")),
                            model_answer= answer,          # ★ model's prediction
                            gold_answer=  correct_answer,  # ★ ground truth
                        )
            agent_answers = {
                agent_id: (
                    agent.outputs[-1]
                    if isinstance(agent.outputs, list) and agent.outputs
                    else agent.outputs
                )
                for agent_id, agent in rg.nodes.items()
            }
            agent_answers["__decision__"] = (
                rg.decision_node.outputs[-1]
                if isinstance(rg.decision_node.outputs, list) and rg.decision_node.outputs
                else rg.decision_node.outputs
            )
            skill_names = {
                agent_id: (
                    agent.current_skill.skill_name
                    if getattr(agent, "current_skill", None) else None
                )
                for agent_id, agent in rg.nodes.items()
            }

            results.append({
                "id":            str(record.get("id", "")),
                "question":      str(record.get("question", "")),
                "predicted":     answer,
                "correct":       correct_answer,
                "is_correct":    is_correct,
                "reason":        reason,
                "raw_answer":    raw_text,
                "skill_names":   skill_names,    
                "agent_answers": agent_answers,
            })

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    accuracy.print()
    print("Done!")
    return accuracy.get(), results
