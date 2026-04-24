# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import io
import os
from typing import Optional

from SkillGraph.skills.skill_library import Skill, SkillLibrary
import json


EVOLVE_PROMPT_TEMPLATE = """
You are an expert AI skill designer for a visual question-answering agent.

Below is a reasoning skill that has been performing poorly, along with its recent failure cases.
Each failure case includes: the question, the answer choices, the agent's wrong answer,
the correct answer, and a diagnostic lesson that pinpoints exactly why the skill failed.

[Current Skill]
Name: {skill_name}
Version: v{version}
Trigger Condition:
{trigger_condition}

Strategy Description:
{description}

[Aggregated Diagnostic Lessons]
{lesson_summary}

[Detailed Failure Cases]
{failure_summary}

[Your Task]
The [Aggregated Diagnostic Lessons] section distills the root causes across the recent failures.
Use these lessons as your PRIMARY basis for deciding what to change and how to change it.

Specifically:
- If the lessons point to failures that can be resolved by refining the current skill,
  choose MODIFY and update both trigger_condition and description so that every lesson is
  clearly addressed.
- If the failure pattern exposes a recurring capability gap that is not adequately covered
  by any existing skill in the library, and cannot be cleanly absorbed by refining the
  current skill, choose CREATE_NEW and synthesize a focused new skill for that gap.

[When You Choose MODIFY]
- Address every lesson above, but do not map lessons one-to-one into new clauses.
- Prefer revising, merging, generalizing, or replacing existing principles over appending
  parallel rules.
- If multiple lessons share a root cause, capture them with one unified fix.
- If an old clause is redundant, overly narrow, or stale, rewrite or remove it.
- Keep the final trigger_condition and description concise, readable, and usable under time pressure.
- Do not make generic improvements with no clear connection to the lessons.

[When You Choose CREATE_NEW]
- Use CREATE_NEW only when the failures reveal a recurring and reusable capability gap,
  not just a one-off weakness in the current skill wording.
- The new skill must be narrow, coherent, and clearly distinguishable from existing skills.
- Its trigger_condition should make its retrieval boundary explicit.
- Its description should focus only on the missing capability revealed by the lessons.
- Do not create a broad catch-all skill or a near-duplicate of an existing one.
- The new skill should be specific enough to retrieve reliably, but broad enough to generalize
  beyond the observed failures.

Respond ONLY with the following JSON (no markdown, no extra text):
{{
  "action": "modify" or "create_new",
  "skill_name": "skill name (same as current for modify, new name for create_new)",
  "trigger_condition": "updated trigger condition",
  "description": "complete updated strategy description",
  "reason": "for each lesson above, one sentence explaining how your change addresses it"
}}
"""

# 单条失败案例格式化模板（含 choices 和 lesson）
_CASE_TEMPLATE = """[Case {idx}] ID: {case_id}
  Question: {question}
  Choices:
{choices_block}
  Agent answered: {model_answer}
  Correct answer: {gold_answer}
  Diagnostic lesson: {lesson}"""


def _format_choices(choices: dict) -> str:
    if not choices:
        return "    (no choices recorded)"
    return "\n".join(f"    {k}: {v}" for k, v in choices.items())


def _build_lesson_summary(skill: Skill, max_cases: int = 5) -> str:
    cases   = skill.failure_cases[-max_cases:]
    lessons = [c.get("lesson", "").strip() for c in cases if c.get("lesson", "").strip()]
    if not lessons:
        return "(no diagnostic lessons available)"
    return "\n".join(f"  {i+1}. {l}" for i, l in enumerate(lessons))


def _build_failure_summary(skill: Skill, max_cases: int = 5) -> str:
    cases = skill.failure_cases[-max_cases:]
    if not cases:
        return "(no failure cases recorded)"
    lines = []
    for idx, c in enumerate(cases, 1):
        lines.append(_CASE_TEMPLATE.format(
            idx           = idx,
            case_id       = c.get("id",           "unknown"),
            question      = c.get("question",     "(no question)"),
            choices_block = _format_choices(c.get("choices", {})),
            model_answer  = c.get("model_answer", "?"),
            gold_answer   = c.get("gold_answer",  "?"),
            lesson        = c.get("lesson",       "(no lesson)"),
        ))
    return "\n\n".join(lines)


def _encode_image_for_llm(image_path: str) -> Optional[dict]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    except Exception as e:
        print(f"[SkillDesigner] image encode failed for {image_path}: {e}")
        return None


class SkillDesigner:
    def __init__(self, skill_library: SkillLibrary, llm):
        self.skill_library = skill_library
        self.llm           = llm

    async def evolve(self, threshold: int = 1):
        hard_skills = self.skill_library.get_hard_cases(threshold=threshold)
        if not hard_skills:
            print("[SkillDesigner] No skills need evolving.")
            return
        for skill in hard_skills:
            print(f"[SkillDesigner] Evolving: {skill.skill_name} "
                  f"(failures={len(skill.failure_cases)}, π={skill.performance_score:.2f})")
            await self._evolve_one(skill)

    async def _evolve_one(self, skill: Skill):
        lesson_summary  = _build_lesson_summary(skill,  max_cases=5)
        failure_summary = _build_failure_summary(skill, max_cases=5)

        text_prompt = EVOLVE_PROMPT_TEMPLATE.format(
            skill_name        = skill.skill_name,
            version           = skill.version,
            trigger_condition = skill.trigger_condition,
            description       = skill.description,
            lesson_summary    = lesson_summary,   # 
            failure_summary   = failure_summary,
        )

        image_block = None
        for case in reversed(skill.failure_cases[-5:]):
            image_path = case.get("image_path", "")
            if image_path:
                image_block = _encode_image_for_llm(image_path)
                if image_block is not None:
                    print(f"[SkillDesigner] attaching image: {image_path}")
                    break

        user_content = (
            [image_block, {"type": "text", "text": text_prompt}]
            if image_block is not None
            else text_prompt
        )

        messages = [{"role": "user", "content": user_content}]

        try:
            response = await self.llm.agen(messages)
        except Exception as e:
            print(f"[SkillDesigner] LLM call failed for {skill.skill_name}: {e}")
            return

        try:
            clean = response.strip()
            if clean.startswith("```"):
                parts = clean.split("```")
                clean = parts[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            result = json.loads(clean.strip())
        except json.JSONDecodeError as e:
            print(f"[SkillDesigner] JSON parse failed for {skill.skill_name}. "
                  f"Raw: {response[:200]}  Error: {e}")
            return

        await self._apply_evolution(skill, result)

    async def _apply_evolution(self, original_skill: Skill, result: dict):
        action = result.get("action", "").lower()

        if action == "modify":
            original_skill.description       = result["description"]
            original_skill.trigger_condition = result.get(
                "trigger_condition", original_skill.trigger_condition
            )
            original_skill.version      += 1
            original_skill.failure_cases = []
            original_skill.source        = "distilled_from_failure"
            self.skill_library.add_or_update_skill(original_skill)
            print(f"[SkillDesigner] ✏️  Modified: {original_skill.skill_name} "
                  f"→ v{original_skill.version}")
            print(f"              Reason: {result.get('reason', '')}")

        elif action == "create_new":
            if self.skill_library.get_skill_by_name(result["skill_name"]) is not None:
                print(f"[SkillDesigner] '{result['skill_name']}' already exists, skipping.")
                return
            new_id = (
                f"{self.skill_library.domain}_"
                f"{result['skill_name'].lower().replace(' ', '_')}_v1"
            )
            new_skill = Skill(
                skill_id          = new_id,
                skill_name        = result["skill_name"],
                scope             = "task-specific",
                trigger_condition = result["trigger_condition"],
                description       = result["description"],
                source            = "distilled_from_failure",
                performance_score = 0.0,
                tools             = list(getattr(original_skill, "tools", []) or []), 
            )
            self.skill_library.add_or_update_skill(new_skill)
            print(f"[SkillDesigner] ✨ Created: {new_skill.skill_name}")
            print(f"              Reason: {result.get('reason', '')}")

        else:
            print(f"[SkillDesigner] Unknown action '{action}', skipping.")
