# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from collections import Counter
from typing import List

from SkillGraph.skills.skill_library import Skill, SkillLibrary


# Aligned with Appendix C (tau_f = 4). The previous default of 1 meant a single
# failure triggered a rewrite, which contradicts the paper's claim that
# evolution is driven by "consistent failure patterns rather than isolated
# anomalies". Pass explicitly from config if you want a different value.
DEFAULT_FAILURE_THRESHOLD = 4


EVOLVE_PROMPT_TEMPLATE = """
You are an expert AI skill designer for a visual question-answering agent.

Below is a reasoning skill that has been performing poorly, along with its recent failure cases.
Each failure case includes: the question, the answer choices, the agent's wrong answer,
the correct answer, a failure type, and a diagnostic lesson that pinpoints why the skill failed.

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

Weight the evidence by failure type:
- PERCEPTION dominant: the skill's observations of the image are unreliable. The fix belongs in
  the strategy: add explicit verification steps (re-examine the relevant region, count twice,
  cross-check each option against a stated observation), or narrow the trigger condition so that
  a tool-equipped skill handles these cases instead.
- REASONING dominant: the skill sees correctly but infers badly. The fix belongs in the inference
  procedure: require the observation to be stated before a conclusion is drawn, require each
  option to be eliminated against a stated observation, forbid concluding from priors alone.
- AMBIGUOUS cases carry the least weight. Do not build the revision around them, and never treat
  an AMBIGUOUS case as evidence of a perception problem.

Then choose one action:
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


_CASE_TEMPLATE = """[Case {idx} | {failure_type}] ID: {case_id}
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
    """
    Aggregate per-case diagnoses into the evidence block for evolution.

    The failure-type histogram is the point of this function: it tells the
    designer whether the skill is failing because it sees badly or because it
    reasons badly, which are opposite fixes. Historical cases written before
    failure_type existed default to AMBIGUOUS.
    """
    cases = skill.failure_cases[-max_cases:]
    if not cases:
        return "(no diagnostic lessons available)"

    counts  = Counter(c.get("failure_type") or "AMBIGUOUS" for c in cases)
    n_blind = sum(1 for c in cases if c.get("image_seen") is False)

    header = (
        f"  Failure-type distribution over the {len(cases)} most recent failures: "
        f"PERCEPTION={counts.get('PERCEPTION', 0)}, "
        f"REASONING={counts.get('REASONING', 0)}, "
        f"AMBIGUOUS={counts.get('AMBIGUOUS', 0)}."
    )
    if n_blind:
        header += (
            f"\n  ({n_blind} of these were diagnosed without access to the image; "
            f"treat their conclusions as weaker evidence.)"
        )

    lines = [header, ""]
    for i, c in enumerate(cases, 1):
        lesson = (c.get("lesson") or "").strip() or "(no lesson recorded)"
        ftype  = c.get("failure_type") or "AMBIGUOUS"
        lines.append(f"  {i}. [{ftype}] {lesson}")
    return "\n".join(lines)


def _build_failure_summary(skill: Skill, max_cases: int = 5) -> str:
    cases = skill.failure_cases[-max_cases:]
    if not cases:
        return "(no failure cases recorded)"
    lines = []
    for idx, c in enumerate(cases, 1):
        lines.append(_CASE_TEMPLATE.format(
            idx           = idx,
            failure_type  = c.get("failure_type") or "AMBIGUOUS",
            case_id       = c.get("id",           "unknown"),
            question      = c.get("question",     "(no question)"),
            choices_block = _format_choices(c.get("choices", {})),
            model_answer  = c.get("model_answer", "?"),
            gold_answer   = c.get("gold_answer",  "?"),
            lesson        = c.get("lesson",       "(no lesson)"),
        ))
    return "\n\n".join(lines)


class SkillDesigner:
    def __init__(self, skill_library: SkillLibrary, llm):
        self.skill_library = skill_library
        self.llm           = llm

    async def evolve(self, threshold: int = DEFAULT_FAILURE_THRESHOLD):
        hard_skills = self.skill_library.get_hard_cases(threshold=threshold)
        if not hard_skills:
            print(f"[SkillDesigner] No skills need evolving (threshold={threshold}).")
            return
        for skill in hard_skills:
            print(f"[SkillDesigner] Evolving: {skill.skill_name} "
                  f"(failures={len(skill.failure_cases)}, π={skill.performance_score:.2f})")
            await self._evolve_one(skill)

    async def _evolve_one(self, skill: Skill):
        lesson_summary  = _build_lesson_summary(skill,  max_cases=5)
        failure_summary = _build_failure_summary(skill, max_cases=5)

        prompt = EVOLVE_PROMPT_TEMPLATE.format(
            skill_name        = skill.skill_name,
            version           = skill.version,
            trigger_condition = skill.trigger_condition,
            description       = skill.description,
            lesson_summary    = lesson_summary,
            failure_summary   = failure_summary,
        )

        # Deliberately text-only.
        #
        # Every lesson in lesson_summary was produced with its own image
        # attached (AnalyzeAgent._generate_lesson), so the visual evidence is
        # already encoded per case. The previous version attached the first
        # loadable image from the batch, which meant case 1's image sat next to
        # a prompt describing five unrelated failures with nothing marking the
        # correspondence - biasing the model toward treating that one scene as
        # the common cause. Cross-failure generalisation is a language-level
        # operation; do it in language.
        messages = [{"role": "user", "content": prompt}]

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
        except (json.JSONDecodeError, IndexError) as e:
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
            print(f"[SkillDesigner] Modified: {original_skill.skill_name} "
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
                parent_id         = original_skill.skill_id,
            )
            self.skill_library.add_or_update_skill(new_skill)
            print(f"[SkillDesigner] Created: {new_skill.skill_name} "
                  f"(parent={original_skill.skill_name})")
            print(f"              Reason: {result.get('reason', '')}")

        else:
            print(f"[SkillDesigner] Unknown action '{action}', skipping.")
