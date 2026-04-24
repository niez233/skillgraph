# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional
import re

from SkillGraph.graph.node import Node
from SkillGraph.agents.agent_registry import AgentRegistry
from SkillGraph.llm.llm_registry import LLMRegistry
from SkillGraph.prompt.prompt_set_registry import PromptSetRegistry
from SkillGraph.tools.search.wiki import search_wiki_main

_LESSON_PROMPT_TEMPLATE = """You are a diagnostic analyst for a visual question-answering agent.

A reasoning agent just answered a multiple-choice question incorrectly.

[Question]
{question}

[Answer Choices]
{choices_str}

[Agent's Full Response (Wrong)]
{model_reasoning}

[Correct Answer]
{gold_answer}

[Active Skill Strategy Used]
{skill_description}

Analyze in 2–3 sentences:
1. What specific reasoning error is visible in the agent's response above?
2. What aspect of the skill's trigger condition or strategy caused this error?

Output only the diagnostic lesson — no preamble, no bullet points, plain text.
"""

_SIM_SWITCH_THRESHOLD = 0.75
_PERF_SWITCH_MARGIN   = 0.10


def find_strings_between_pluses(text: str) -> List[str]:
    return re.findall(r"\@(.*?)\@", text)


def _encode_image_to_content(image_path: str) -> Optional[dict]:
    if not image_path or not os.path.exists(str(image_path)):
        print(f"[Image] path not found or empty: {image_path}")
        return None
    try:
        from PIL import Image as PILImage
        img  = PILImage.open(image_path).convert("RGB")
        w, h = img.size
        print(f"[Image] original size: {w}x{h}")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    except ImportError:
        print("[Image] PIL not installed")
        return None
    except Exception as e:
        print(f"[Image] encode failed: {e}")
        return None


def _build_choices_str(choices: Optional[dict]) -> str:
    if not choices:
        return "(no choices provided)"
    return "\n".join(f"  {k}: {v}" for k, v in choices.items())


def _build_answer_with_text(letter: str, choices: Optional[dict]) -> str:
    if choices and letter in choices:
        return f"{letter}: {choices[letter]}"
    return letter


@AgentRegistry.register("AnalyzeAgent")
class AnalyzeAgent(Node):
    def __init__(
        self,
        id: Optional[str]   = None,
        role: Optional[str] = None,
        domain: str         = "",
        llm_name: str       = "",
        skill_library       = None,
        constraint_suffix: str = ""
    ):
        super().__init__(id, "AnalyzeAgent", domain, llm_name)

        self.llm           = LLMRegistry.get(llm_name)
        self.prompt_set    = PromptSetRegistry.get(domain)
        self.skill_library = skill_library
        self.current_skill = None

        if self.skill_library is not None:
            available = self.skill_library.get_all_skills()
            if not available:
                raise ValueError("skill_library.get_all_skills() returned empty list.")
            idx                = hash(id or "") % len(available)
            self.current_skill = available[idx]
            self.role          = self.current_skill.skill_name
        else:
            self.role = self.prompt_set.get_role() if role is None else role

        self.wiki_summary   = ""
        self._last_response = None
        self.constraint_suffix = constraint_suffix 

    # ------------------------------------------------------------------
    # Dynamic skill selection
    # ------------------------------------------------------------------

    def _select_skill_for_query(
        self,
        query_text: str,
        shared_matches: Optional[list] = None,
        agent_rank:     Optional[int]  = None,
    ) -> None:
        if self.skill_library is None or self.current_skill is None:
            return

        if shared_matches is not None and agent_rank is not None:
            rank               = agent_rank % len(shared_matches)
            skill, sim         = shared_matches[rank]
            self.current_skill = skill
            self.role          = skill.skill_name
            print(f"[Agent {self.id}] shared rank={rank} -> {skill.skill_name}")
            return

        query          = query_text or ""
        current_latest = self.skill_library.get_skill_by_name(self.current_skill.skill_name)
        if current_latest is not None:
            self.current_skill = current_latest

        top_matches = self.skill_library.get_skills_by_query(query, top_k=5)
        if not top_matches:
            return

        best_skill, best_sim = top_matches[0]
        if best_skill.skill_name == self.current_skill.skill_name:
            return

        sim_ok    = best_sim >= _SIM_SWITCH_THRESHOLD
        is_new    = best_skill.usage_count == 0
        is_mature = best_skill.usage_count >= 3
        perf_ok   = (
            best_skill.performance_score
            >= self.current_skill.performance_score + _PERF_SWITCH_MARGIN
        )

        if sim_ok and is_new:
            self.current_skill = best_skill
            self.role          = best_skill.skill_name
            print(f"[Agent {self.id}] trial new skill -> {best_skill.skill_name} (sim={best_sim:.3f})")
        elif sim_ok and is_mature and perf_ok:
            self.current_skill = best_skill
            self.role          = best_skill.skill_name
            print(f"[Agent {self.id}] skill switch -> {best_skill.skill_name} "
                  f"(sim={best_sim:.3f}, perf={best_skill.performance_score:.2f})")

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        if self.current_skill is not None:
            return self.current_skill.to_system_prompt() + self.constraint_suffix 
        return self.prompt_set.get_analyze_constraint(self.role)

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    async def _process_inputs(
        self,
        raw_inputs:    Dict[str, str],
        spatial_info:  Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ):
        system_prompt = self._get_system_prompt()

        if self.role != "Fake":
            user_prompt = f"The task is: {raw_inputs['task']}\n"
        else:
            user_prompt = self.prompt_set.get_adversarial_answer_prompt(raw_inputs["task"])

        spatial_str = ""

        for agent_id, info in spatial_info.items():
            if self.role == "Wiki Searcher" and info["role"] == "Knowlegable Expert":
                queries = find_strings_between_pluses(info["output"])
                wiki    = await search_wiki_main(queries)
                if wiki:
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += (
                        "The key entities of the problem are explained in "
                        f"Wikipedia as follows:{self.wiki_summary}"
                    )
            spatial_str += (
                f"Agent {agent_id}, role is {info['role']}, "
                f"output is:\n\n{info['output']}\n\n"
            )


        if spatial_str:
            user_prompt += (
                f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str}\n\n"
            )

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Sync execute
    # ------------------------------------------------------------------

    def _execute(
        self,
        input:         Dict[str, str],
        spatial_info:  Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs,
    ):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            system_prompt, user_prompt = loop.run_until_complete(
                self._process_inputs(input, spatial_info, temporal_info)
            )
        finally:
            loop.close()
        return self.llm.gen([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ])

    # ------------------------------------------------------------------
    # Async execute
    # ------------------------------------------------------------------

    async def _async_execute(
        self,
        input:          Dict[str, str],
        spatial_info:   Dict[str, Dict],
        temporal_info:  Dict[str, Dict],
        shared_matches: Optional[list] = None,
        agent_rank:     Optional[int]  = None,
        **kwargs,
    ):
        self._select_skill_for_query(
            input.get("task", ""),
            shared_matches=shared_matches,
            agent_rank=agent_rank,
        )

        system_prompt, user_prompt = await self._process_inputs(
            input, spatial_info, temporal_info
        )

        image_block = _encode_image_to_content(input.get("image"))

        print(f"[DEBUG] image_block is None: {image_block is None}")
        if image_block is not None:
            url = image_block.get("image_url", {}).get("url", "")
            print(f"[DEBUG] image_block url prefix: {url[:80]}")

        user_content = (
            [image_block, {"type": "text", "text": user_prompt}]
            if image_block is not None
            else user_prompt
        )

        print(f"[DEBUG] user_content type: {type(user_content)}")
        if isinstance(user_content, list):
            print(f"[DEBUG] user_content[0] keys: {list(user_content[0].keys())}")
        print(f"[DEBUG] messages[1]['content'] type: {type(user_content)}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        response = await self.llm.agen(messages)

        if self.wiki_summary:
            response         += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""

        print(f"################ system prompt: {system_prompt}")
        print(f"################ user prompt:   {user_prompt}")
        print(f"################ response:      {response}")
        return response

    # ------------------------------------------------------------------
    # Lesson 生成：调 LLM 分析单次失败根因
    # ------------------------------------------------------------------

    async def _generate_lesson(
        self,
        question_text: str,
        choices:       Optional[dict],
        model_answer:  str,
        gold_answer:   str,
        model_reasoning: str = "",        # ← 新增
    ) -> str:
        choices_str = _build_choices_str(choices)
        skill_desc  = (
            self.current_skill.description.strip()
            if self.current_skill is not None
            else "(no skill active)"
        )
        prompt = _LESSON_PROMPT_TEMPLATE.format(
            question          = question_text or "",
            choices_str       = choices_str,
            model_reasoning   = model_reasoning or model_answer or "",
            model_answer      = model_answer  or "",
            gold_answer       = gold_answer   or "",
            skill_description = skill_desc,
        )
        try:
            lesson = await self.llm.agen([{"role": "user", "content": prompt}])
            return lesson.strip()
        except Exception as e:
            print(f"[AnalyzeAgent] lesson generation failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Performance recording（async，含 choices / image_path / lesson）
    #
    # 调用方改为：
    #   await agent.record_skill_result(
    #       is_correct    = pred == item["answer"],
    #       question_id   = item["id"],
    #       question_text = item["question"],
    #       choices       = item["choices"],
    #       image_path    = item.get("image", ""),
    #       model_answer  = pred,
    #       gold_answer   = item["answer"],
    #   )
    # ------------------------------------------------------------------

    async def record_skill_result(
        self,
        is_correct:    bool,
        question_id:   Optional[str]  = None,
        question_text: Optional[str]  = None,
        choices:       Optional[dict] = None,
        image_path:    Optional[str]  = None,
        model_answer:  Optional[str]  = None,
        gold_answer:   Optional[str]  = None,
        model_reasoning: Optional[str]  = None,
    ) -> None:
        if self.skill_library is None or self.current_skill is None:
            return

        model_answer_full = _build_answer_with_text(model_answer or "", choices)
        gold_answer_full  = _build_answer_with_text(gold_answer  or "", choices)

        lesson = ""
        if not is_correct and question_id:
            print(f"[AnalyzeAgent {self.id}] generating lesson for {question_id} ...")
            lesson = await self._generate_lesson(
                question_text = question_text or "",
                choices       = choices,
                model_answer  = model_answer_full,
                gold_answer   = gold_answer_full,
                model_reasoning = model_reasoning or "", 
            )
            print(f"[AnalyzeAgent {self.id}] lesson: {lesson[:120]}")

        self.skill_library.update_skill_performance(
            skill_name    = self.current_skill.skill_name,
            is_correct    = is_correct,
            question_id   = question_id,
            question_text = question_text,
            choices       = choices,
            image_path    = image_path,
            model_answer  = model_answer_full,
            gold_answer   = gold_answer_full,
            lesson        = lesson,
            model_reasoning = model_reasoning or "", 
        )
