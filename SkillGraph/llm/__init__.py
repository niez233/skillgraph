from SkillGraph.llm.llm_registry import LLMRegistry

# Backward-compatible alias (older code expects VisualLLMRegistry)
VisualLLMRegistry = LLMRegistry

__all__ = ["LLMRegistry", "VisualLLMRegistry"]
