# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional, Tuple
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

[Agent's Answer]
{model_answer}

[Correct Answer]
{gold_answer}

[Active Skill Strategy Used]
{skill_description}

{image_notice}

Your task has two parts.

PART 1 - Classify this failure into exactly one category:
  PERCEPTION : the agent's description of the image is factually inaccurate
               (miscounted, misread text, missed or hallucinated an object,
                wrong colour / position / size / attribute).
  REASONING  : the agent's description of the image is accurate, but the
               inference drawn from it is flawed (ignored its own observation,
                let a prior override visible evidence, faulty elimination,
                misread the scope of the question).
  AMBIGUOUS  : you cannot determine which of the above applies from the
               evidence available to you.

PART 2 - In at most 60 words, state the single most specific correctable
mistake, and what the skill's trigger condition or strategy should do
differently next time. Do not restate the question. Do not speculate about
image content you cannot verify.

Respond in exactly this format, with no other text:
FAILURE_TYPE: <PERCEPTION|REASONING|AMBIGUOUS>
LESSON: <your diagnosis in 60 words or fewer>
"""

_LESSON_IMAGE_NOTICE_WITH = (
    "[Note] The original image is attached above. Verify the agent's visual "
    "claims against it directly before classifying. If the agent described the "
    "image inaccurately, that is a PERCEPTION failure regardless of how sound "
    "the rest of its reasoning looks."
)

_LESSON_IMAGE_NOTICE_WITHOUT = (
    "[Note] The image is NOT available to you. You can only see the agent's own "
    "description of it, which may itself be wrong. If the failure hinges on "
    "whether that description is accurate, classify it as AMBIGUOUS rather than "
    "guessing."
)

_VALID_FAILURE_TYPES = ("PERCEPTION", "REASONING", "AMBIGUOUS")


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


def _parse_lesson_response(raw: str, image_seen: bool) -> Tuple[str, str]:
    """
    Parse 'FAILURE_TYPE: X\\nLESSON: ...' into (failure_type, lesson).
    Degrades gracefully if the model drifts from the format.
    """
    raw = (raw or "").strip()
    if not raw:
        return "AMBIGUOUS", ""

    failure_type = "AMBIGUOUS"
    lesson       = raw

    m = re.search(r"FAILURE_TYPE\s*[:：]\s*(PERCEPTION|REASONING|AMBIGUOUS)",
                  raw, re.IGNORECASE)
    if m:
        failure_type = m.group(1).upper()

    m = re.search(r"LESSON\s*[:：]\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        lesson = m.group(1).strip()

    # A diagnostician that never saw the image has no grounds to assert that the
    # agent's description of it was wrong. Downgrade rather than trust it.
    if not image_seen and failure_type == "PERCEPTION":
        failure_type = "AMBIGUOUS"

    if failure_type not in _VALID_FAILURE_TYPES:
        failure_type = "AMBIGUOUS"

    return failure_type, lesson


@AgentRegistry.register("AnalyzeAgent")
class AnalyzeAgent(Node):
    def __init__(
        self,
        id: Optional[str]   = None,
        role: Optional[str] = None,
        domain: str         = "",
        llm_name: str       = "",
        skill_library       = None,
        constraint_suffix: str = "",
        agent_index: Optional[int] = None,
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
            # Deterministic assignment. hash() on str is salted per process
            # (PYTHONHASHSEED), which made the initial skill assignment - and
            # therefore role_adj_matrix - vary across runs and occasionally
            # assign the same skill to two agents.
            if agent_index is not None:
                idx = agent_index % len(available)
            else:
                idx = (sum(ord(c) for c in str(id or "")) % len(available))
            self.current_skill = available[idx]
            self.role          = self.current_skill.skill_name
        else:
            self.role = self.prompt_set.get_role() if role is None else role

        self.wiki_summary      = ""
        self._last_response    = None
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

        # ---- Live path -------------------------------------------------
        # graph.arun always supplies shared_matches + agent_rank when a skill
        # library exists, so this branch handles every real call.
        if shared_matches is not None and agent_rank is not None:
            rank               = agent_rank % len(shared_matches)
            skill, sim         = shared_matches[rank]
            self.current_skill = skill
            self.role          = skill.skill_name
            print(f"[Agent {self.id}] shared rank={rank} -> {skill.skill_name}")
            return

        # ---- Unreachable in the current pipeline -----------------------
        # Everything below implements an explore/exploit switching policy
        # (trial-new -> observe -> switch on performance with hysteresis).
        # It is never executed because of the early return above. Two known
        # defects if it is ever re-enabled:
        #   (a) a skill whose usage_count is 1 or 2 satisfies neither is_new
        #       nor is_mature, so it can never be selected again - a deadlock;
        #   (b) current_skill is reset to the initial skill on every sample by
        #       the deepcopy in train_vl.py, so the hysteresis margin compares
        #       against a constant and never actually suppresses a switch.
        # Either wire this in and fix both, or delete it.
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

        user_content = (
            [image_block, {"type": "text", "text": user_prompt}]
            if image_block is not None
            else user_prompt
        )

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
    # Lesson generation: diagnose a single failure, with the image attached
    # ------------------------------------------------------------------

    async def _generate_lesson(
        self,
        question_text:   str,
        choices:         Optional[dict],
        model_answer:    str,
        gold_answer:     str,
        model_reasoning: str = "",
        image_path:      Optional[str] = None,
    ) -> Tuple[str, str, bool]:
        """
        Returns (failure_type, lesson, image_seen).

        The image is attached when it can be loaded. This is the only point in
        the pipeline where the diagnostician can independently check the agent's
        visual claims, which is what separates a perception failure from a
        reasoning failure. Without it the diagnostician inherits the agent's
        (possibly wrong) description and can only speculate.
        """
        choices_str = _build_choices_str(choices)
        skill_desc  = (
            self.current_skill.description.strip()
            if self.current_skill is not None
            else "(no skill active)"
        )

        image_block = _encode_image_to_content(image_path)
        image_seen  = image_block is not None

        prompt = _LESSON_PROMPT_TEMPLATE.format(
            question          = question_text or "",
            choices_str       = choices_str,
            model_reasoning   = model_reasoning or model_answer or "",
            model_answer      = model_answer or "",
            gold_answer       = gold_answer or "",
            skill_description = skill_desc,
            image_notice      = (
                _LESSON_IMAGE_NOTICE_WITH if image_seen
                else _LESSON_IMAGE_NOTICE_WITHOUT
            ),
        )

        content = (
            [image_block, {"type": "text", "text": prompt}]
            if image_seen else prompt
        )

        try:
            raw = await self.llm.agen([{"role": "user", "content": content}])
        except Exception as e:
            print(f"[AnalyzeAgent] lesson generation failed: {e}")
            return "AMBIGUOUS", "", image_seen

        failure_type, lesson = _parse_lesson_response(raw, image_seen=image_seen)
        return failure_type, lesson, image_seen


    async def record_skill_result(
        self,
        is_correct:      bool,
        question_id:     Optional[str]  = None,
        question_text:   Optional[str]  = None,
        choices:         Optional[dict] = None,
        image_path:      Optional[str]  = None,
        model_answer:    Optional[str]  = None,
        gold_answer:     Optional[str]  = None,
        model_reasoning: Optional[str]  = None,
    ) -> None:
        if self.skill_library is None or self.current_skill is None:
            return

        model_answer_full = _build_answer_with_text(model_answer or "", choices)
        gold_answer_full  = _build_answer_with_text(gold_answer  or "", choices)

        lesson       = ""
        failure_type = ""
        image_seen   = False

        if not is_correct and question_id:
            print(f"[AnalyzeAgent {self.id}] generating lesson for {question_id} ...")
            failure_type, lesson, image_seen = await self._generate_lesson(
                question_text   = question_text or "",
                choices         = choices,
                model_answer    = model_answer_full,
                gold_answer     = gold_answer_full,
                model_reasoning = model_reasoning or "",
                image_path      = image_path,
            )
            print(f"[AnalyzeAgent {self.id}] [{failure_type}] "
                  f"(image_seen={image_seen}) {lesson[:120]}")

        self.skill_library.update_skill_performance(
            skill_name      = self.current_skill.skill_name,
            is_correct      = is_correct,
            question_id     = question_id,
            question_text   = question_text,
            choices         = choices,
            image_path      = image_path,
            model_answer    = model_answer_full,
            gold_answer     = gold_answer_full,
            lesson          = lesson,
            model_reasoning = model_reasoning or "",
            failure_type    = failure_type,   # NEW - needs skill_library support
            image_seen      = image_seen,     # NEW - for auditing
        )

