from SkillGraph.utils.globals import Cost, PromptTokens, CompletionTokens
import tiktoken
import json

# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing


def _to_text(x) -> str:
    """Convert any object to a safe string for tokenization."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # common: OpenAI SDK style objects with `.content`
    if hasattr(x, "content"):
        c = getattr(x, "content", None)
        if isinstance(c, str):
            return c
    # dict/list/etc -> json
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def cal_token(model: str, text) -> int:
    """Count tokens safely; text may not be str."""
    safe_text = _to_text(text)

    # tiktoken may raise if model unknown; fallback to a base encoding.
    try:
        encoder = tiktoken.encoding_for_model(model)
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")

    num_tokens = len(encoder.encode(safe_text))
    return num_tokens


def cost_count(prompt, response, model_name):
    branch: str
    prompt_len: int
    completion_len: int
    price: float

    prompt_len = cal_token(model_name, prompt)
    completion_len = cal_token(model_name, response)

    if "gpt-4" in model_name:
        branch = "gpt-4"
        price = (
            prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] / 1000
            + completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] / 1000
        )
    elif "gpt-3.5" in model_name:
        branch = "gpt-3.5"
        price = (
            prompt_len * OPENAI_MODEL_INFO[branch][model_name]["input"] / 1000
            + completion_len * OPENAI_MODEL_INFO[branch][model_name]["output"] / 1000
        )
    elif "dall-e" in model_name:
        branch = "dall-e"
        price = 0.0
        prompt_len = 0
        completion_len = 0
    else:
        branch = "other"
        price = 0.0
        prompt_len = 0
        completion_len = 0

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    return price, prompt_len, completion_len


OPENAI_MODEL_INFO = {
    "gpt-4": {
        "current_recommended": "gpt-4-1106-preview",
        "gpt-4-0125-preview": {
            "context window": 128000,
            "training": "Jan 2024",
            "input": 0.01,
            "output": 0.03,
        },
        "gpt-4-1106-preview": {
            "context window": 128000,
            "training": "Apr 2023",
            "input": 0.01,
            "output": 0.03,
        },
        "gpt-4-vision-preview": {
            "context window": 128000,
            "training": "Apr 2023",
            "input": 0.01,
            "output": 0.03,
        },
        "gpt-4": {
            "context window": 8192,
            "training": "Sep 2021",
            "input": 0.03,
            "output": 0.06,
        },
        "gpt-4-0314": {
            "context window": 8192,
            "training": "Sep 2021",
            "input": 0.03,
            "output": 0.06,
        },
        "gpt-4-32k": {
            "context window": 32768,
            "training": "Sep 2021",
            "input": 0.06,
            "output": 0.12,
        },
        "gpt-4-32k-0314": {
            "context window": 32768,
            "training": "Sep 2021",
            "input": 0.06,
            "output": 0.12,
        },
        "gpt-4-0613": {
            "context window": 8192,
            "training": "Sep 2021",
            "input": 0.06,
            "output": 0.12,
        },
        "gpt-4o": {
            "context window": 128000,
            "training": "Jan 2024",
            "input": 0.005,
            "output": 0.015,
        },
    },
    "gpt-3.5": {
        "current_recommended": "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125": {
            "context window": 16385,
            "training": "Jan 2024",
            "input": 0.0010,
            "output": 0.0020,
        },
        "gpt-3.5-turbo-1106": {
            "context window": 16385,
            "training": "Sep 2021",
            "input": 0.0010,
            "output": 0.0020,
        },
        "gpt-3.5-turbo-instruct": {
            "context window": 4096,
            "training": "Sep 2021",
            "input": 0.0015,
            "output": 0.0020,
        },
        "gpt-3.5-turbo": {
            "context window": 4096,
            "training": "Sep 2021",
            "input": 0.0015,
            "output": 0.0020,
        },
        "gpt-3.5-turbo-0301": {
            "context window": 4096,
            "training": "Sep 2021",
            "input": 0.0015,
            "output": 0.0020,
        },
        "gpt-3.5-turbo-0613": {
            "context window": 16384,
            "training": "Sep 2021",
            "input": 0.0015,
            "output": 0.0020,
        },
        "gpt-3.5-turbo-16k-0613": {
            "context window": 16384,
            "training": "Sep 2021",
            "input": 0.0015,
            "output": 0.0020,
        },
    },
    "dall-e": {
        "current_recommended": "dall-e-3",
        "dall-e-3": {
            "release": "Nov 2023",
            "standard": {
                "1024×1024": 0.040,
                "1024×1792": 0.080,
                "1792×1024": 0.080,
            },
            "hd": {
                "1024×1024": 0.080,
                "1024×1792": 0.120,
                "1792×1024": 0.120,
            },
        },
        "dall-e-2": {
            "release": "Nov 2022",
            "1024×1024": 0.020,
            "512×512": 0.018,
            "256×256": 0.016,
        },
    },
}
