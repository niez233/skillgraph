# -*- coding: utf-8 -*-
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from datetime import datetime
import inspect
import textwrap

@dataclass
class Skill:
    skill_id: str
    skill_name: str
    scope: str
    trigger_condition: str
    description: str
    source: str = "human_designed"
    parent_id: Optional[str] = None
    version: int = 1
    performance_score: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    # 完整失败记录：id / question / choices / image_path / model_answer / gold_answer / lesson
    failure_cases: List[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tools: List[str] = field(default_factory=list) 

    def update_performance(
        self,
        is_correct:    bool,
        question_id:   str  = None,
        question_text: str  = None,
        choices:       dict = None,
        image_path:    str  = None,
        model_answer:  str  = None,
        gold_answer:   str  = None,
        lesson:        str  = None,
        model_reasoning: str  = None, 
    ):
        self.usage_count += 1
        if is_correct:
            self.success_count += 1
        else:
            if question_id:
                record = {
                    "id":           question_id,
                    "question":     question_text or "",
                    "choices":      choices       or {},
                    "image_path":   image_path    or "",
                    "model_reasoning": model_reasoning or "",  
                    "model_answer": model_answer  or "",
                    "gold_answer":  gold_answer   or "",
                    "lesson":       lesson        or "",
                }
                self.failure_cases.append(record)
                if len(self.failure_cases) > 20:
                    self.failure_cases = self.failure_cases[-20:]
        self.performance_score = self.success_count / self.usage_count
        self.updated_at = datetime.now().isoformat()

    def to_system_prompt(self) -> str:
        base = (
            f"[Reasoning Strategy: {self.skill_name} v{self.version}]\n\n"
            f"## When to Apply\n{self.trigger_condition}\n\n"
            f"{self.description}\n"
        )
        tools = getattr(self, "tools", None) or []
        if not tools:
            return base

        from SkillGraph.tools.vision_tools import _TOOL_FUNCTIONS
        tool_snippets = []
        for tname in tools:
            fn = _TOOL_FUNCTIONS.get(tname)
            if fn is None:
                continue
            try:
                src = textwrap.dedent(inspect.getsource(fn))
            except (OSError, TypeError):
                continue
            tool_snippets.append(f"### {tname}\n```python\n{src}```")

        if not tool_snippets:
            return base
        
        tools_block = (
    "\n## Tool Reference\n"
    "The following tool snippets are for reference only. "
    "Use them as helpful guidance when reasoning about the image, but do not follow them strictly. "
    "Do not include code in your answer.\n\n"
    + "\n\n".join(tool_snippets)
    + "\n"
)


        return base + tools_block



    def to_retrieval_text(self) -> str:
        return f"[Trigger] {self.trigger_condition}\n[Strategy] {self.description[:300]}"


INITIAL_SKILLS_MMBENCH = [
    Skill(
        skill_id="mmbench_visual_analyst",
        skill_name="Visual Analyst",
        scope="task-specific",
        parent_id=None,
        trigger_condition="When a problem requires overall image perception but cannot be clearly categorized as OCR or pure object perception, this serves as a general fallback",
        description="""
You are an expert at analyzing images in detail.
Before answering, describe the overall scene: identify salient objects, their attributes,
spatial layout, and any text present.
Strategy: When uncertain which aspect of the image is most relevant,
enumerate all visible elements first, then decide which are pertinent to the question.
Avoid: Do not skip straight to the answer — always ground your response in image evidence.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_ocr",
        skill_name="OCR Specialist",
        scope="task-specific",
        parent_id="mmbench_visual_analyst",
        trigger_condition="When the task involves recognizing text, numbers, symbols, labels, signs, or handwritten content in an image, OCR Specialist is activated first",
        description="""
You are an expert at reading and interpreting text within images (OCR).
Carefully locate all text regions before answering: signs, labels, captions,
handwriting, watermarks, and embedded numbers all count.
Strategy: Read character by character for ambiguous glyphs; pay attention to
font orientation (rotated or mirrored text), partial occlusion, and low contrast.
Avoid: Do not infer what text "should" say based on context alone —
read exactly what is written, even if it seems unusual.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_perception",
        skill_name="Perception Specialist",
        scope="task-specific",
        parent_id="mmbench_visual_analyst",
        trigger_condition="When the task involves identifying object category, color, shape, material, state, or scene type, it takes precedence over Visual Analyst",
        description="""
You are an expert at visual perception: recognizing objects, their categories,
attributes (color, shape, texture, material), and the overall scene type.
Strategy: Start with coarse categorization,
then progressively identify objects from foreground to background.
For each object note: category → color → shape → state (e.g. open/closed, broken/intact).
Avoid: Do not conflate visually similar categories without checking distinguishing features
(e.g. leopard vs. cheetah, mug vs. cup).
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_spatial_reasoner",
        skill_name="Spatial Reasoner",
        scope="task-specific",
        parent_id=None,
        trigger_condition="When the task involves spatial relationships, positions, quantities, or geometric reasoning but cannot be clearly classified as pure localization or counting, this serves as a general fallback",
        description="""
You are an expert at spatial and geometric reasoning.
Analyze the image for positional relationships, relative sizes, orientations,
and quantities of objects.
Strategy: Establish a consistent coordinate frame (e.g. viewer's left/right)
before making any directional claim. Cross-check counts and positions
against multiple reference objects to avoid errors.
Avoid: Do not rely on gestalt impression for either location or quantity —
be explicit and systematic.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_location",
        skill_name="Location Specialist",
        scope="task-specific",
        parent_id="mmbench_spatial_reasoner",
        trigger_condition="When the task involves determining object position, direction, relative relationships (above/below/left/right/front/back/inside/outside) or absolute regions (corners, center), activate before Spatial Reasoner",
        description="""
You are an expert at determining the location and spatial relationships of objects in images.
Strategy: Always anchor to a fixed reference object first, then describe other objects
relative to it using consistent terms (viewer's left/right, not the subject's).
For region-based questions, mentally divide the image into a 3×3 grid
and identify which cell(s) each object occupies.
Avoid: Do not use vague terms like "nearby" or "around" — be precise
(e.g. "directly to the left of X", "in the upper-right quadrant").
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_counting",
        skill_name="Counting Specialist",
        scope="task-specific",
        parent_id="mmbench_spatial_reasoner",
        trigger_condition="When the task involves counting the number of a certain type of object in an image, activate before Spatial Reasoner",
        description="""
You are an expert at accurately counting objects in images.
Strategy: Scan the image in a fixed raster order (left-to-right, top-to-bottom),
mentally marking each counted instance to avoid double-counting or omission.
For cluttered scenes, group objects into sub-regions and sum the totals.
Verify by doing a second pass with a different scanning direction.
Avoid: Do not estimate by impression — always count explicitly.
Watch for partially occluded objects that still represent a full instance.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_critic",
        skill_name="Critic",
        scope="general",
        parent_id=None,
        trigger_condition="When other agents' analyses need to be questioned and verified",
        description="""
You are an excellent critic.
Point out potential issues in other agents' analysis point by point.
Check whether the answer is consistent with what is actually visible in the image.
Strategy: For each claim made by other agents, ask "Is this directly visible in the image?"
Lesson from past failures: Do not just agree with majority — images often contain subtle details
that contradict obvious interpretations.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_domain_expert",
        skill_name="Domain Expert",
        scope="task-specific",
        parent_id=None,
        trigger_condition="When the task involves professional domain knowledge such as science, history, culture, or everyday common sense",
        description="""
You are a knowledgeable expert across multiple domains including science, history, culture, and everyday life.
Use your domain knowledge to reason about the image content and answer correctly.
Strategy: First identify the domain the question belongs to, then recall relevant knowledge
to interpret visual cues that a general observer might misread.
Avoid: Do not over-rely on domain knowledge alone — always verify your interpretation
against what is actually visible in the image.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_logician",
        skill_name="Logician",
        scope="general",
        parent_id=None,
        trigger_condition="When the task requires step-by-step reasoning, elimination of wrong options, or logical deduction",
        description="""
You are an expert at logical reasoning and deduction.
Given the visual evidence and the question, reason step by step to eliminate wrong options
and identify the correct answer.
Strategy: List all options explicitly, then eliminate each with a concrete reason grounded
in image evidence until only one remains.
Avoid: Do not jump to conclusions — every elimination must be justified.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_commonsense_reasoner",
        skill_name="Commonsense Reasoner",
        scope="task-specific",
        parent_id=None,
        trigger_condition="When the task involves everyday scenes, human behavior, object uses, or real-world regularities",
        description="""
You are strong at commonsense reasoning about real-world scenes and everyday situations.
Use what you know about how the world works to interpret the image and answer the question.
Strategy: Ask yourself "What would normally happen in this scene?" and cross-check
that expectation with the actual visual content.
Avoid: Do not let commonsense assumptions override direct image evidence —
the image may depict an unusual or staged scenario.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_detail_inspector",
        skill_name="Detail Inspector",
        scope="task-specific",
        parent_id=None,
        trigger_condition="When the task involves colors, textures, text, numbers, small objects, or other easily overlooked details",
        description="""
You are meticulous and pay close attention to fine-grained details.
Look carefully at colors, textures, small objects, numbers, and subtle cues in the image
that others might miss.
Strategy: Mentally zoom into different regions of the image one by one;
do not stop at the most salient object.
Avoid: Do not dismiss small or peripheral elements — the correct answer often hinges
on a detail that is easy to overlook at first glance.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_contextual_reasoner",
        skill_name="Contextual Reasoner",
        scope="task-specific",
        parent_id=None,
        trigger_condition="When the task requires inference combining the overall image context, scene background, or contextual clues",
        description="""
You are good at understanding context and inferring meaning from partial information.
Use surrounding context in the image to reason about what is happening and select the best answer.
Strategy: Consider the overall scene setting first, then interpret individual elements
in light of that context rather than in isolation.
Avoid: Do not fixate on a single object out of context —
meaning often emerges from the relationship between elements.
""",
        source="human_designed",
    ),
    Skill(
        skill_id="mmbench_skeptic",
        skill_name="Skeptic",
        scope="general",
        parent_id=None,
        trigger_condition="When the task contains misleading options or the most obvious answer may be a trap",
        description="""
You are a skeptic who questions assumptions.
Challenge the most obvious-looking answer and verify it against the actual image evidence
before committing.
Strategy: Deliberately consider why the seemingly correct answer might be wrong,
then look for image evidence that either confirms or refutes it.
Lesson from past failures: The distractor option is often designed to match the first
impression — slow down and double-check before finalizing.
""",
        source="human_designed",
    ),
]


class SkillLibrary:
    def __init__(self, domain: str, save_path: str = None):
        self.domain    = domain
        self.save_path = save_path or f"./skill_library_{domain}.json"
        self.skills: dict[str, Skill] = {}
        self._skill_embeddings          = None
        self._skill_id_order: List[str] = []
        self._emb_dirty: bool           = True

        if os.path.exists(self.save_path):
            self.load()
        else:
            for skill in INITIAL_SKILLS_MMBENCH:
                self.skills[skill.skill_id] = skill
            self.save()

    def get_all_skills(self) -> List[Skill]:
        return list(self.skills.values())

    def get_skill_by_name(self, name: str) -> Optional[Skill]:
        for skill in self.skills.values():
            if skill.skill_name == name:
                return skill
        return None

    def get_top_k_skills(self, k: int, exclude_names: List[str] = None) -> List[Skill]:
        candidates = [s for s in self.skills.values()
                      if not exclude_names or s.skill_name not in exclude_names]
        candidates.sort(key=lambda s: s.performance_score, reverse=True)
        return candidates[:k]

    def get_skills_by_query(self, query: str, top_k: int = 3) -> List[Tuple[Skill, float]]:
        import numpy as np
        from SkillGraph.llm.profile_embedding import get_sentence_embedding

        skills = list(self.skills.values())
        if not skills:
            return []
        if self._emb_dirty or len(self._skill_id_order) != len(skills):
            self._rebuild_embeddings(skills)

        q_emb = np.array(get_sentence_embedding(query), dtype=np.float32)
        norm  = np.linalg.norm(q_emb)
        if norm > 1e-8:
            q_emb /= norm

        sims    = self._skill_embeddings @ q_emb
        k       = min(top_k, len(skills))
        top_idx = np.argsort(sims)[::-1][:k]
        id_to_skill = {s.skill_id: s for s in skills}
        return [
            (id_to_skill[self._skill_id_order[i]], float(sims[i]))
            for i in top_idx
            if self._skill_id_order[i] in id_to_skill
        ]

    def _rebuild_embeddings(self, skills: List[Skill]):
        import numpy as np
        from SkillGraph.llm.profile_embedding import get_sentence_embedding

        self._skill_id_order = [s.skill_id for s in skills]
        embs = []
        for skill in skills:
            emb  = np.array(get_sentence_embedding(skill.to_retrieval_text()), dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb /= norm
            embs.append(emb)
        self._skill_embeddings = np.stack(embs)
        self._emb_dirty        = False
        print(f"[SkillLibrary] embedding cache rebuilt, skills={len(skills)}")

    def update_skill_performance(
        self,
        skill_name:    str,
        is_correct:    bool,
        question_id:   str  = None,
        question_text: str  = None,
        choices:       dict = None,
        image_path:    str  = None,
        model_answer:  str  = None,
        gold_answer:   str  = None,
        lesson:        str  = None,
        model_reasoning: str  = None,  
    ):
        skill = self.get_skill_by_name(skill_name)
        if skill:
            skill.update_performance(
                is_correct, question_id, question_text,
                choices, image_path, model_answer, gold_answer, lesson, model_reasoning
            )
            self.save()

    def add_or_update_skill(self, skill: Skill):
        self.skills[skill.skill_id] = skill
        self._emb_dirty = True
        self.save()

    def get_hard_cases(self, threshold: int = 3) -> List[Skill]:
        return [s for s in self.skills.values() if len(s.failure_cases) >= threshold]

    def save(self):
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(
                {sid: asdict(s) for sid, s in self.skills.items()},
                f, ensure_ascii=False, indent=2,
            )

    def load(self):
        with open(self.save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.skills     = {sid: Skill(**s) for sid, s in data.items()}
        self._emb_dirty = True

_SKILL_TOOL_MAP = {
    "OCR Specialist":        ["run_ocr"],
    "Counting Specialist":   ["detect_objects"],
    "Perception Specialist": ["detect_objects", "color_histogram"],
    "Location Specialist":   ["overlay_grid", "detect_objects"],
    "Detail Inspector":      ["color_histogram"],
    "Visual Analyst":        ["detect_objects"],
    "Spatial Reasoner":      ["overlay_grid", "detect_objects"],
}


def attach_tools_to_library(lib: SkillLibrary) -> None:
    changed = False
    for name, tools in _SKILL_TOOL_MAP.items():
        s = lib.get_skill_by_name(name)
        if s is not None and not s.tools:
            s.tools = list(tools)
            changed = True
    if changed:
        lib._emb_dirty = True
        lib.save()
        print(f"[SkillLibrary] attached tools to initial skills.")
