import aiohttp
from typing import List, Union, Optional, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
import json

from SkillGraph.llm.format import Message
from SkillGraph.llm.price import cost_count
from SkillGraph.llm.llm import LLM

# -----------------------------
# Load env (.env) reliably
# -----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ENV_PATH = os.path.join(_REPO_ROOT, ".env")
load_dotenv(dotenv_path=_ENV_PATH)

MINE_BASE_URL = os.getenv("BASE_URL")   # e.g. http://127.0.0.1:8001
MINE_API_KEYS = os.getenv("API_KEY")   # e.g. local

# Your server only exposes this model id (from /v1/models)
DEFAULT_SERVER_MODEL_ID = "/public/niezheng/models/Qwen3-VL-8B-Instruct"

# Map common names (like gpt-4o) to the real served model id
MODEL_ALIAS: Dict[str, str] = {
    "gpt-4o": DEFAULT_SERVER_MODEL_ID,
    "gpt-4": DEFAULT_SERVER_MODEL_ID,
    "gpt-3.5-turbo": DEFAULT_SERVER_MODEL_ID,
    "default": DEFAULT_SERVER_MODEL_ID,
}


def _safe_str(x) -> str:
    """Convert any object to a printable string without crashing."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "content") and isinstance(getattr(x, "content"), str):
        return x.content
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _extract_text_from_response(response_data: Any) -> str:
    """
    Extract plain assistant text from OpenAI-compatible response:
      {"choices":[{"message":{"content":"..."}}], ...}
    Fallback: stringify.
    """
    if response_data is None:
        return ""

    if isinstance(response_data, str):
        return response_data

    if isinstance(response_data, dict):
        # OpenAI / vLLM chat-completions style
        if "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
            c0 = response_data["choices"][0]
            if isinstance(c0, dict):
                if "message" in c0 and isinstance(c0["message"], dict):
                    return _safe_str(c0["message"].get("content", ""))
                if "text" in c0:
                    return _safe_str(c0["text"])
            return _safe_str(c0)

        # Some servers return {"data": "..."}
        if "data" in response_data:
            return _safe_str(response_data["data"])

        # Error payload
        if "error" in response_data:
            return _safe_str(response_data["error"])

        return _safe_str(response_data)

    return _safe_str(response_data)


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
):
    """
    Call OpenAI-compatible endpoint: POST /v1/chat/completions
    """
    base = (MINE_BASE_URL or "").rstrip("/")
    request_url = base + "/v1/chat/completions"

    if not base:
        raise ValueError(f"BASE_URL not set. Check {_ENV_PATH} or export BASE_URL.")
    if not request_url.strip():
        raise ValueError(f"Invalid request_url: {request_url!r}")

    # Map model name to server model id
    real_model = MODEL_ALIAS.get(model, model)
    # If user passes some random string, but server only has one model, force it:
    if real_model != DEFAULT_SERVER_MODEL_ID and real_model.startswith("gpt-"):
        real_model = DEFAULT_SERVER_MODEL_ID

    headers = {"Content-Type": "application/json"}
    # Standard bearer header (works for most OpenAI-compatible servers)
    if MINE_API_KEYS:
        headers["Authorization"] = f"Bearer {MINE_API_KEYS}"

    data = {
        "model": real_model,
        "messages": msg,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for m in msg:
        if m["role"] == "user":
            print(f"[DEBUG] user content type sent to server: {type(m['content'])}")
            if isinstance(m["content"], list):
                print(f"[DEBUG] user content blocks: {[b.get('type') for b in m['content']]}")

    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            raw_text = await response.text()
            try:
                response_data = json.loads(raw_text)
            except Exception:
                response_data = raw_text

            # DEBUG (you can comment out later)
            print("DEBUG request_url:", request_url)
            print("DEBUG model:", real_model)
            print("DEBUG status:", response.status)
            # print("DEBUG payload(messages)[0:1]:", msg[:1])
            if isinstance(response_data, dict):
                print("DEBUG response keys:", list(response_data.keys())[:20])
            else:
                print("DEBUG response text head:", str(response_data)[:300])

            prompt = "".join([_safe_str(item.get("content", "")) for item in msg])
            output_text = _extract_text_from_response(response_data)

            # If server returned an error, raise to trigger retry (and make it obvious)
            if isinstance(response_data, dict) and "error" in response_data:
                # still count tokens for visibility
                cost_count(prompt, output_text, model)
                raise RuntimeError(f"LLM server error: {response_data['error']}")

            cost_count(prompt, output_text, model)
            return output_text


class GPTChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        # Allow passing a plain string
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        # Convert possible inputs to dicts for achat()
        msg: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, dict):
                if "role" in m and "content" in m:
                    raw_content = m["content"]
                    # list 说明是多模态消息（图片+文字），直接保留，不转字符串
                    content = raw_content if isinstance(raw_content, list) else _safe_str(raw_content)
                    msg.append({"role": _safe_str(m["role"]), "content": content})
                elif "task" in m:
                    msg.append({"role": "user", "content": _safe_str(m["task"])})
                else:
                    msg.append({"role": "user", "content": _safe_str(m)})
            else:
                msg.append({
                    "role": _safe_str(getattr(m, "role", "user")),
                    "content": _safe_str(getattr(m, "content", "")),
                })
        return await achat(self.model_name, msg, max_tokens=max_tokens, temperature=temperature)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        raise NotImplementedError("GPTChat.gen is not implemented; use agen() instead.")


def _register_gptchat():
    from SkillGraph.llm.llm_registry import LLMRegistry
    LLMRegistry.register("GPTChat")(GPTChat)


_register_gptchat()
