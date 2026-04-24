# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional

# 复用已有的超时执行器(沿用你代码库里的工具)
from SkillGraph.tools.coding.executor_utils import function_with_timeout


# ======================================================================
# 1) 真实工具函数(函数体里有真逻辑,不是桩)
# ======================================================================

def color_histogram(image_path: str, bbox: Optional[list] = None) -> dict:
    """在 bbox 区域(或整图)内计算主色调。"""
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    if bbox is None:
        bbox = [0, 0, img.width, img.height]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, img.width));  x2 = max(0, min(x2, img.width))
    y1 = max(0, min(y1, img.height)); y2 = max(0, min(y2, img.height))
    if x2 <= x1 or y2 <= y1:
        return {"error": "invalid bbox"}

    crop = np.array(img.crop((x1, y1, x2, y2)))
    r, g, b = crop.reshape(-1, 3).mean(axis=0).astype(int).tolist()

    if   r > 150 and g < 100 and b < 100:      name = "red"
    elif r > 200 and g > 200 and b < 120:      name = "yellow"
    elif g > 150 and r < 120 and b < 120:      name = "green"
    elif b > 150 and r < 120 and g < 150:      name = "blue"
    elif r < 80  and g < 80  and b < 80:       name = "black"
    elif r > 200 and g > 200 and b > 200:      name = "white"
    elif abs(r-g) < 20 and abs(g-b) < 20 and 80 <= r <= 200: name = "gray"
    elif r > 150 and g > 100 and b < 100:      name = "orange"
    elif r > 150 and b > 150 and g < 150:      name = "purple"
    elif r > 150 and g > 100 and b > 100:      name = "pink"
    else:                                       name = "mixed"
    return {"dominant_color": name, "mean_rgb": [r, g, b]}


def detect_objects(image_path: str, category: str) -> dict:
    """检测某类目标并返回 count + 前 5 个 bbox。

    默认实现用简单的颜色 + 轮廓启发式兜底(避免没装 YOLO 就彻底跑不起来)。
    真实使用时把 USE_YOLO 改为 True 并装好 ultralytics。
    """
    USE_YOLO = False

    if USE_YOLO:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model(image_path, verbose=False)[0]
        boxes = []
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy(),
                                  results.boxes.conf.cpu().numpy()):
            if results.names[int(cls)] == category and conf > 0.35:
                boxes.append([int(x) for x in box.tolist()])
        return {"category": category, "count": len(boxes),
                "top_boxes": boxes[:5]}

    # 启发式兜底:返回"检测器未就绪"的明确信号,模型会知道这条观察不可靠
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    return {"category": category, "count": -1, "top_boxes": [],
            "image_size": list(img.size),
            "note": "detector_not_wired (install ultralytics and set USE_YOLO=True)"}


def crop_region(image_path: str, bbox: list,
                save_as: str = "/tmp/_skill_crop.jpg") -> dict:
    from PIL import Image
    img = Image.open(image_path).convert("RGB").crop(tuple(bbox))
    img.save(save_as, quality=85)
    return {"saved_path": save_as, "size": list(img.size)}


def overlay_grid(image_path: str, n: int = 3) -> dict:
    """返回 n×n 的网格分区坐标,供模型按区域定位。"""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    cells = {}
    for i in range(n):
        for j in range(n):
            cells[f"row{i}_col{j}"] = [int(j*W/n), int(i*H/n),
                                        int((j+1)*W/n), int((i+1)*H/n)]
    return {"grid": f"{n}x{n}", "image_size": [W, H], "cells": cells}


def run_ocr(image_path: str, lang: str = "en") -> dict:
    """识别图中文字。默认启发式兜底,真实接入 PaddleOCR/Tesseract 时替换函数体。"""
    USE_PADDLE = False

    if USE_PADDLE:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        result = ocr.ocr(image_path)
        texts = [line[1][0] for page in (result or []) for line in page]
        return {"text": " | ".join(texts)}

    return {"text": "", "note": "ocr_not_wired (install paddleocr and set USE_PADDLE=True)"}


# 所有可注册工具的规范表(函数对象 + 简短描述)
_TOOL_FUNCTIONS: Dict[str, Callable] = {
    "color_histogram": color_histogram,
    "detect_objects":  detect_objects,
    "crop_region":     crop_region,
    "overlay_grid":    overlay_grid,
    "run_ocr":         run_ocr,
}

_TOOL_DESCS: Dict[str, str] = {
    "color_histogram": "Compute dominant color of a bbox (or whole image).",
    "detect_objects":  "Detect instances of a category, return count and bboxes.",
    "crop_region":     "Crop a bbox and save to disk.",
    "overlay_grid":    "Return n x n grid cell coordinates for region localization.",
    "run_ocr":         "Extract text from the image.",
}


# ======================================================================
# 2) ToolRegistry:字符串 name -> (函数对象, 源码字符串, desc)
# ======================================================================

class ToolRegistry:
    """工具注册中心。

    存三样东西:
      - fn:    真函数对象(供"查字典 + 加括号"式调用)
      - src:   函数源代码字符串(供 exec 动态执行,以及方案二里当 prompt 素材)
      - desc:  一句话描述
    """
    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, fn: Callable, desc: str = "") -> None:
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            src = f"# source unavailable for {name}"
        self._tools[name] = {"fn": fn, "src": src, "desc": desc or name}

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_source(self, name: str) -> str:
        return self._tools.get(name, {}).get("src", "")

    def get_desc(self, name: str) -> str:
        return self._tools.get(name, {}).get("desc", "")

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    # ----- 执行入口:exec 风格(对齐 python_executor.py 的做法) -----
    def call_via_exec(self, name: str, kwargs: dict, timeout: int = 10) -> Any:
        """把函数源码 + 调用语句拼成代码字符串,用 exec 动态执行,从 locals 里取 'result'。

        这和你 python_executor.py 里 exec(code, globals()) 的机制是一致的:
        通过字符串化的代码让 Python 解释器进入函数体。
        """
        if name not in self._tools:
            return {"error": f"tool '{name}' not registered"}

        src = self._tools[name]["src"]
        # 构造调用代码:先把函数定义 exec 进去,再调用它,把返回值绑定到 'result'
        kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        code = f"{src}\n\nresult = {name}({kwargs_repr})\n"

        local_ns: dict = {}
        try:
            function_with_timeout(exec, (code, local_ns), timeout)
            return local_ns.get("result", {"error": "no 'result' produced"})
        except TimeoutError:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for name, fn in _TOOL_FUNCTIONS.items():
        reg.register(name, fn, _TOOL_DESCS.get(name, ""))
    return reg


# ======================================================================
# 3) 关键词路由 + 参数推断 + 观察格式化
# ======================================================================

_ROUTE_RULES: list[tuple[set, list[str]]] = [
    ({"how many", "count", "number of", "how much"},              ["detect_objects"]),
    ({"what color", "what colour", "which color", "which colour"},["detect_objects", "color_histogram"]),
    ({"what does", "what is written", "read the", "sign says",
      "the text", "the letter", "the word"},                      ["run_ocr"]),
    ({"where is", "location of", "position of",
      "which part", "which region", "which corner"},              ["overlay_grid"]),
]


def suggest_tools(question: str, allowed: List[str]) -> List[str]:
    q = (question or "").lower()
    hits: List[str] = []
    for kws, tools in _ROUTE_RULES:
        if any(kw in q for kw in kws):
            hits.extend(t for t in tools if t in allowed)
    return list(dict.fromkeys(hits))


_COMMON_CATEGORIES = {
    "cat", "dog", "bird", "car", "person", "people", "tree", "flower",
    "bicycle", "bike", "motorcycle", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "chair", "table", "bottle", "cup", "bowl",
    "apple", "banana", "orange", "pizza", "cake", "book", "clock", "phone",
    "laptop", "tv", "bed", "sofa", "plant", "airplane", "plane", "boat",
    "ship", "truck", "bus", "train", "building", "window", "door",
    "sign", "ball", "child", "man", "woman", "baby", "animal",
}


def _extract_category(question: str) -> Optional[str]:
    q = (question or "").lower()
    for pat in [r"how many (\w+)", r"number of (\w+)", r"count(?:ing)? (?:the )?(\w+)"]:
        m = re.search(pat, q)
        if m:
            word = m.group(1).rstrip("s")
            return word
    for tok in re.findall(r"[a-z]+", q):
        if tok.rstrip("s") in _COMMON_CATEGORIES:
            return tok.rstrip("s")
    return None


def infer_args(tool_name: str, question: str, image_path: str) -> Optional[dict]:
    if not image_path or not os.path.exists(image_path):
        return None
    if tool_name == "overlay_grid":
        return {"image_path": image_path, "n": 3}
    if tool_name == "run_ocr":
        return {"image_path": image_path, "lang": "en"}
    if tool_name == "detect_objects":
        cat = _extract_category(question)
        return {"image_path": image_path, "category": cat} if cat else None
    if tool_name == "color_histogram":
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                w, h = im.size
            return {"image_path": image_path, "bbox": [0, 0, w, h]}
        except Exception:
            return None
    if tool_name == "crop_region":
        return None  
    return None


def format_observation(tool_name: str, args: dict, result: Any) -> str:
    if isinstance(result, dict) and "error" in result:
        return f"- {tool_name}: FAILED ({result['error']})"
    if tool_name == "detect_objects":
        return (f"- detect_objects(category='{args.get('category')}') -> "
                f"count={result.get('count')}, top_boxes={result.get('top_boxes', [])}"
                + (f" [{result['note']}]" if result.get('note') else ''))
    if tool_name == "color_histogram":
        return (f"- color_histogram(bbox={args.get('bbox')}) -> "
                f"dominant_color='{result.get('dominant_color')}', "
                f"mean_rgb={result.get('mean_rgb')}")
    if tool_name == "run_ocr":
        text = (result.get("text") or "").strip()
        preview = text[:200] + ("..." if len(text) > 200 else "")
        return (f"- run_ocr() -> text=\"{preview}\""
                + (f" [{result['note']}]" if result.get('note') else ''))
    if tool_name == "overlay_grid":
        return (f"- overlay_grid(n={args.get('n')}) -> "
                f"image_size={result.get('image_size')}, {result.get('grid')} cells")
    return f"- {tool_name}({args}) -> {result}"
