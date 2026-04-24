from typing import Union, Dict, Any, List
import itertools

from SkillGraph.prompt.prompt_set import PromptSet
from SkillGraph.prompt.prompt_set_registry import PromptSetRegistry
from SkillGraph.prompt.common import get_combine_materials


roles = itertools.cycle([
    'Visual Analyst',
    'Critic',
    'Spatial Reasoner',
    'Domain Expert',
    'Logician',
    'Commonsense Reasoner',
    'Detail Inspector',
    'Contextual Reasoner',
    'Skeptic',
])


ROLE_DESCRIPTION = {
"Visual Analyst":
"""
You are an expert at analyzing images in detail.
Describe what you see in the image carefully before answering: identify objects, their attributes, spatial relationships, and any text visible.
""",
"Critic":
"""
You are an excellent critic.
Point out potential issues in other agents' analysis point by point.
Check whether the answer is consistent with what is actually visible in the image.
""",
"Spatial Reasoner":
"""
You are an expert at spatial and geometric reasoning.
You excel at understanding positions, directions, sizes, quantities, and relationships between objects in an image.
""",
"Domain Expert":
"""
You are a knowledgeable expert across multiple domains including science, history, culture, and everyday life.
Use your domain knowledge to reason about the image content and answer correctly.
""",
"Logician":
"""
You are an expert at logical reasoning and deduction.
Given the visual evidence and the question, reason step by step to determine whether the statement about the image is true or false.
""",
"Commonsense Reasoner":
"""
You are strong at commonsense reasoning about real-world scenes and everyday situations.
Use what you know about how the world works to interpret the image and judge whether the statement is correct.
""",
"Detail Inspector":
"""
You are meticulous and pay close attention to fine-grained details.
Look carefully at colors, textures, small objects, numbers, and subtle cues in the image that others might miss.
""",
"Contextual Reasoner":
"""
You are good at understanding context and inferring meaning from partial information.
Use surrounding context in the image to reason about what is happening and judge the statement.
""",
"Skeptic":
"""
You are a skeptic who questions assumptions.
Challenge the most obvious-looking answer and verify it against the actual image evidence before committing.
""",
"Fake":
"""
You are a liar who only tells lies.
""",
}

ROLE_CONNECTION = [
    ('Visual Analyst',      'Critic'),
    ('Visual Analyst',      'Logician'),
    ('Visual Analyst',      'Spatial Reasoner'),
    ('Visual Analyst',      'Domain Expert'),
    ('Spatial Reasoner',    'Logician'),
    ('Spatial Reasoner',    'Critic'),
    ('Domain Expert',       'Logician'),
    ('Domain Expert',       'Critic'),
    ('Logician',            'Critic'),
    ('Logician',            'Skeptic'),
    ('Commonsense Reasoner','Critic'),
    ('Commonsense Reasoner','Logician'),
    ('Detail Inspector',    'Visual Analyst'),
    ('Detail Inspector',    'Critic'),
    ('Contextual Reasoner', 'Domain Expert'),
    ('Contextual Reasoner', 'Logician'),
    ('Skeptic',             'Critic'),
    ('Skeptic',             'Visual Analyst'),
    ('Critic',              'Commonsense Reasoner'),
    ('Critic',              'Contextual Reasoner'),
    ('Logician',            'Detail Inspector'),
    ('Domain Expert',       'Commonsense Reasoner'),
    ('Spatial Reasoner',    'Detail Inspector'),
    ('Commonsense Reasoner','Spatial Reasoner'),
    ('Contextual Reasoner', 'Visual Analyst'),
    ('Detail Inspector',    'Domain Expert'),
    ('Skeptic',             'Spatial Reasoner'),
    ('Critic',              'Logician'),
    ('Visual Analyst',      'Commonsense Reasoner'),
    ('Domain Expert',       'Detail Inspector'),
]


@PromptSetRegistry.register('mme')
class MMEPromptSet(PromptSet):
    """
    MME prompt set for binary (Yes / No) visual question answering.
    Each question is accompanied by an image; the answer is Yes or No.
    """

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return (
            "You are the top decision-maker who is skilled at analyzing and summarizing "
            "other agents' visual observations and reasoning, identifying errors, "
            "and giving the final correct Yes or No answer."
        )

    def get_role_connection(self):
        return ROLE_CONNECTION

    def get_description(self, role):
        return ROLE_DESCRIPTION.get(role, "")

    @staticmethod
    def get_constraint():
        return """
I will show you an image and ask you a yes/no question about it.
You must answer with exactly one word: Yes or No.
Your answer can refer to the answers of other agents provided to you.
Your reply must be less than 100 words but include your answer and a brief step-by-step analysis.
The first line of your reply must contain only one word: Yes or No.
"""

    @staticmethod
    def get_analyze_constraint(role):
        role_desc = ROLE_DESCRIPTION.get(role, "")
        return role_desc + """
I will show you an image and ask you a yes/no question about it.
Using the reasoning from other agents as additional advice with critical thinking, give an updated answer.
You are strictly prohibited from imitating the analysis process of other agents.
Your reply must be less than 100 words but include your answer and a brief step-by-step analysis.
The first line of your reply must contain only one word: Yes or No.
"""

    @staticmethod
    def get_decision_constraint():
        return """
I will show you an image and ask you a yes/no question about it.
I will give you some other agents' answers and analysis.
Your reply must only contain one word (Yes or No) and cannot have any other characters.
For example, your reply can be: Yes
"""

    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return (
            f"Give a wrong answer and false analysis for the following visual yes/no question: {question}.\n"
            "No matter what you see in the image, please only output lies and try your best to mislead other agents.\n"
            "Your reply must be less than 100 words.\n"
            "The first line of your reply must contain only one word: Yes or No."
        )

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_decision_few_shot():
        return ""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        # Normalise to lowercase "yes" / "no"
        t = answer.strip().lower()
        ans_pos = t.find("answer is")
        if ans_pos != -1:
            t = t[ans_pos + len("answer is"):].strip(": ").strip()
        if t.startswith("yes"):
            return "yes"
        if t.startswith("no"):
            return "no"
        # Fallback: scan for yes/no word
        for word in t.split():
            if word in ("yes", "no"):
                return word
        return t[:3] if t else ""
