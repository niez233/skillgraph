"""
Filename: mmbench_dataset.py

MMBench Dataset wrapper for SkillGraph.
Mirrors the MMLUDataset interface so it plugs directly into
experiments/train_vl.py and experiments/evaluate_vl.py.

Data source (priority):
  1. Explicit `data_jsonl` path (from vmas_local_prepare.py output)
  2. Auto-resolve under `out_root/mmbench/{dataset_id}__{split}__*/data.jsonl`
  3. Download from HuggingFace if neither is available
      (requires `datasets` + `Pillow` + `jsonlines`)

MMBench answer format: A / B / C / D  (multiple choice)
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class MMBenchDataset:
    """
    MMBench visual multiple-:
        id       : str
        image    : str   (absolute path to image file)
        question : str
        choices  : dict  e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}
        answer   : str   e.g. "A"
        meta     : dict

    Compatible interface:
        __len__, __getitem__,
        record_to_input(record) -> {"task": str, "image": str}
        postprocess_answer(answer: str | list) -> str
        record_to_target_answer(record) -> str
    """

    def __init__(self, split: str='train', data_jsonl: Optional[str]=None, out_root: str='./datasets_out', dataset_id: str='HuggingFaceM4/MMBench_dev', limit: Optional[int]=None, hf_endpoint: Optional[str]=None, auto_download: bool=True) -> None:
        self._split = split
        self._dataset_id = dataset_id
        self._data: List[Dict[str, Any]] = self._load_data(data_jsonl=data_jsonl, out_root=out_root, dataset_id=dataset_id, split=split, limit=limit, hf_endpoint=hf_endpoint, auto_download=auto_download)
        print(f'[MMBenchDataset] Loaded {len(self._data)} samples (split={split})')

    @staticmethod
    def _safe_dataset_id(s: str) -> str:
        return s.replace('/', '_').replace(':', '_').replace(' ', '_')

    def _load_data(self, data_jsonl: Optional[str], out_root: str, dataset_id: str, split: str, limit: Optional[int], hf_endpoint: Optional[str], auto_download: bool) -> List[Dict[str, Any]]:
        if data_jsonl:
            return self._read_jsonl(Path(data_jsonl), limit)
        base = Path(out_root) / 'mmbench'
        if base.exists():
            safe_id = self._safe_dataset_id(dataset_id)
            cands = sorted(list(base.glob(f'{safe_id}__{split}__*')), key=lambda p: p.stat().st_mtime, reverse=True)
            for c in cands:
                jp = c / 'data.jsonl'
                if jp.exists():
                    print(f'[MMBenchDataset] Auto-resolved: {jp}')
                    return self._read_jsonl(jp, limit)
        if auto_download:
            print(f'[MMBenchDataset] No local data found; downloading from HuggingFace ({dataset_id})...')
            return self._download_and_cache(out_root=out_root, dataset_id=dataset_id, split=split, limit=limit, hf_endpoint=hf_endpoint)
        raise FileNotFoundError(f'[MMBenchDataset] Cannot find prepared dataset.\n  Looked under: {base}\n  Run vmas_local_prepare.py first, or pass data_jsonl=... explicitly.')

    @staticmethod
    def _read_jsonl(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
                if limit and len(data) >= limit:
                    break
        return data

    @staticmethod
    def _safe_str(x: Any) -> str:
        if x is None:
            return ''
        s = str(x)
        return '' if s.lower() in ('nan', 'none') else s

    @staticmethod
    def _infer_answer(row: Dict[str, Any]) -> str:
        """Derive the letter answer (A-H) from various HuggingFace column names."""
        if row.get('answer') and MMBenchDataset._safe_str(row['answer']):
            return MMBenchDataset._safe_str(row['answer']).strip().upper()
        lab = row.get('label')
        if lab is not None:
            if isinstance(lab, int):
                mapping = 'ABCDEFGH'
                return mapping[lab] if 0 <= lab < len(mapping) else str(lab)
            s = MMBenchDataset._safe_str(lab).strip()
            m = re.search('\\b([A-H])\\b', s.upper())
            return m.group(1) if m else s
        return ''

    def _download_and_cache(self, out_root: str, dataset_id: str, split: str, limit: Optional[int], hf_endpoint: Optional[str]) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset
            import jsonlines
        except ImportError as e:
            raise RuntimeError('Auto-download requires: pip install datasets pillow jsonlines') from e
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint
        safe_id = self._safe_dataset_id(dataset_id)
        tag = f'{safe_id}__{split}' + (f'__limit{limit}' if limit else '')
        base = Path(out_root) / 'mmbench' / tag
        base.mkdir(parents=True, exist_ok=True)
        images_dir = base / 'images'
        images_dir.mkdir(exist_ok=True)
        jsonl_path = base / 'data.jsonl'
        if jsonl_path.exists() and any(images_dir.iterdir()):
            return self._read_jsonl(jsonl_path, limit)
        ds = load_dataset(dataset_id, split=split)
        records: List[Dict[str, Any]] = []
        with jsonlines.open(jsonl_path, 'w') as w:
            for (i, item) in enumerate(ds):
                if limit and i >= limit:
                    break
                row = dict(item)
                q = self._safe_str(row.get('question'))
                hint = self._safe_str(row.get('hint'))
                if hint:
                    q = f'{q}\n\nHint:\n{hint}'
                choices: Dict[str, str] = {}
                for key in list('ABCDEFGH'):
                    vs = self._safe_str(row.get(key))
                    if vs:
                        choices[key] = vs
                ans = self._infer_answer(row).upper()
                ex_id = self._safe_str(row.get('index') or row.get('id')) or str(i)
                img_path = images_dir / f'{ex_id}.png'
                _save_any_image(row.get('image'), img_path)
                rec = {'id': f'mmbench:{ex_id}', 'image': str(img_path.resolve()), 'question': q, 'choices': choices if choices else None, 'answer': ans, 'meta': {'source': 'MMBench', 'dataset_id': dataset_id, 'split': split}}
                w.write(rec)
                records.append(rec)
        return records

    @staticmethod
    def get_domain() -> str:
        return 'mmbench'

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._data[index]

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a record as a SkillGraph input dict.
        Returns {"task": <question_with_choices>, "image": <abs_image_path>}.
        """
        q = record['question']
        choices = record.get('choices') or {}
        if isinstance(choices, dict):
            choice_lines = '\n'.join((f'Option {k}: {v}' for (k, v) in sorted(choices.items())))
        else:
            choice_lines = ''
        if choice_lines:
            option_letters = ', '.join(sorted(choices.keys()))   # 动态生成选项列表
            task = f'{q}\n{choice_lines}\n\nWhich option is correct?'
        else:
            task = f'{q}\n\nWhich option is correct?'
        return {'task': task, 'image': str(record.get('image') or '')}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ''
        if not isinstance(answer, str):
            raise Exception(f'Expected string, got {type(answer)}')
        if not answer:
            return ''

        # 1. 优先匹配 "answer is X" / "answer: X"
        m = re.search(r'\bthe\s+answer\s+is[:\s]+([A-D])', answer, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 2. 匹配行首孤立字母（模型按格式输出时）
        m = re.search(r'(?:^|\n)\s*([A-D])\s*(?:\n|$|[.\)])', answer)
        if m:
            return m.group(1).upper()

        # 3. fallback：找任意 A-D 字母
        m = re.search(r'\b([A-D])\b', answer.upper())
        if m:
            return m.group(1)

        return answer[0].upper()
    
    def get_constraint_suffix(self) -> str:
        return """
I will show you an image and ask you a multiple-choice question.
Only one answer is correct.
Using the reasoning from other agents as additional advice with critical thinking, give an updated answer.
You are strictly prohibited from imitating the analysis process of other agents.
Your reply must be less than 100 words but include your answer and a brief step-by-step analysis.
The first must contain only one letter (for example: A, B, C or D).
Your response MUST follow this exact format (no deviation):
<LETTER>
<your reasoning in <= 80 words>
"""

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        ans = str(record.get('answer') or '').strip().upper()
        assert isinstance(ans, str), f'String expected but got {ans!r} of type {type(ans)}; record={record}'
        return ans

def _save_any_image(obj: Any, out_path: Path) -> None:
    """Save image object (PIL.Image, dict with bytes/path, or str path) to out_path."""
    try:
        from PIL import Image as PILImage
    except ImportError as e:
        raise RuntimeError('Pillow required: pip install pillow') from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, PILImage.Image):
        obj.save(out_path, format='PNG')
        return
    if isinstance(obj, dict):
        if obj.get('bytes') is not None:
            out_path.write_bytes(obj['bytes'])
            return
        if obj.get('path'):
            PILImage.open(obj['path']).convert('RGB').save(out_path, format='PNG')
            return
    if isinstance(obj, str) and obj:
        PILImage.open(obj).convert('RGB').save(out_path, format='PNG')
        return
    raise ValueError(f'Unsupported image object type: {type(obj)}')

