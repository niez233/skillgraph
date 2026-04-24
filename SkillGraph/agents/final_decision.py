from typing import List, Any, Dict, Optional

from SkillGraph.graph.node import Node
from SkillGraph.agents.agent_registry import AgentRegistry
from SkillGraph.llm.llm_registry import LLMRegistry
from SkillGraph.prompt.prompt_set_registry import PromptSetRegistry
from SkillGraph.tools.coding.python_executor import PyExecutor

# 复用 AnalyzeAgent 里的图像编码函数，避免重复实现
from SkillGraph.agents.analyze_agent import _encode_image_to_content


@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "FinalWriteCode", domain, llm_name)
        self.llm        = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines  = (line.strip() for line in prompt.split('\n') if line.strip())
        results     = []
        lines_iter  = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call   = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")
        return results

    def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ) -> List[Any]:
        self.role       = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt   = f"{self.role}.\n {self.constraint}"
        spatial_str     = ""
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```"):
                self.internal_tests = self.extract_example(raw_inputs)
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(
                    output, self.internal_tests, timeout=10
                )
                spatial_str += (
                    f"Agent {id} as a {info['role']}:\n\n"
                    f"The code written by the agent is:\n\n{info['output']}\n\n"
                    f"Whether it passes internal testing? {is_solved}.\n\n"
                    f"The feedback is:\n\n{feedback}.\n\n"
                )
            else:
                spatial_str += (
                    f"Agent {id} as a {info['role']} provides the following info: "
                    f"{info['output']}\n\n"
                )
        user_prompt = (
            f"The task is:\n\n{raw_inputs['task']}.\n"
            f"At the same time, the outputs and feedbacks of other agents are as follows:"
            f"\n\n{spatial_str}\n\n"
        )
        return system_prompt, user_prompt

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message  = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ]
        return self.llm.gen(message)

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message  = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ]
        return await self.llm.agen(message)


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "FinalRefer", domain, llm_name)
        self.llm        = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ) -> List[Any]:
        self.role       = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt   = f"{self.role}.\n {self.constraint}"

        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} The task is:\n\n {raw_inputs['task']}.\n"
            f"At the same time, the output of other agents is as follows:\n\n{spatial_str}"
        )
        return system_prompt, user_prompt

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ]
        return self.llm.gen(message)

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)

        # FinalRefer 也需要看图，才能在各 agent 答案有分歧时做出正确裁决
        # image_block = _encode_image_to_content(input.get("image"))

        # if image_block is not None:
        #     user_content = [
        #         image_block,
        #         {"type": "text", "text": user_prompt},
        #     ]
        # else:
        #     user_content = user_prompt
        user_content = user_prompt

        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_content},
        ]

        response = await self.llm.agen(message)
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response


@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = ""):
        """Used for Directed IO"""
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        return None

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        info_list = [info['output'] for info in spatial_info.values()]
        return info_list[-1] if info_list else ""

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        info_list = [info['output'] for info in spatial_info.values()]
        return info_list[-1] if info_list else ""


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = ""):
        """Used for Directed IO"""
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        return None

    def _vote(self, spatial_info: Dict[str, Any]) -> str:
        output_num   = {}
        max_output   = ""
        max_count    = 0
        for info in spatial_info.values():
            ans = self.prompt_set.postprocess_answer(info['output'])
            output_num[ans] = output_num.get(ans, 0) + 1
            if output_num[ans] > max_count:
                max_output = ans
                max_count  = output_num[ans]
        return max_output

    def _execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        return self._vote(spatial_info)

    async def _async_execute(
        self,
        input: Dict[str, str],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        **kwargs,
    ):
        result = self._vote(spatial_info)
        print(result)
        return result
