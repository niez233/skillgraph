"""
Filename: mme_dataset.py

MME Dataset wrapper for SkillGraph.
Mirrors the MMLUDataset interface so it plugs directly into
experiments/train_vl.py and experiments/evaluate_vl.py.

Data source (priority):
  1. Explicit `data_jsonl` path (from vmas_local_prepare.py output)
  2. Auto-resolve under `out_root/mme/{dataset_id}__{split}__*/data.jsonl`
  3. Download from HuggingFace if neither is available
      (requires `datasets` + `Pillow` + `jsonlines`)

MME answer format: "Yes" / "No"
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class MMEDataset:
    """
    MME visual QA dataset.

    Each record (dict) exposes:
        id       : str
        image    : str  (absolute path to image file)
        question : str
        choices  : None  (MME is Yes/No, no explicit choices)
        answer   : str   ("yes" / "no")
        meta     : dict

    Compatible interface:
        __len__, __getitem__,
        record_to_input(record) -> {"task": str, "image": str}
        postprocess_answer(answer: str | list) -> str
        record_to_target_answer(record) -> str
    """

    def __init__(self, split: str='test', data_jsonl: Optional[str]=None, out_root: str='./datasets_out', dataset_id: str='lmms-lab/MME', limit: Optional[int]=None, hf_endpoint: Optional[str]=None, auto_download: bool=True) -> None:
        self._split = split
        self._dataset_id = dataset_id
        self._data: List[Dict[str, Any]] = self._load_data(data_jsonl=data_jsonl, out_root=out_root, dataset_id=dataset_id, split=split, limit=limit, hf_endpoint=hf_endpoint, auto_download=auto_download)
        print(f'[MMEDataset] Loaded {len(self._data)} samples (split={split})')

    @staticmethod
    def _safe_dataset_id(s: str) -> str:
        return s.replace('/', '_').replace(':', '_').replace(' ', '_')

    def _load_data(self, data_jsonl: Optional[str], out_root: str, dataset_id: str, split: str, limit: Optional[int], hf_endpoint: Optional[str], auto_download: bool) -> List[Dict[str, Any]]:
        if data_jsonl:
            return self._read_jsonl(Path(data_jsonl), limit)
        base = Path(out_root) / 'mme'
        if base.exists():
            safe_id = self._safe_dataset_id(dataset_id)
            cands = sorted(list(base.glob(f'{safe_id}__{split}__*')), key=lambda p: p.stat().st_mtime, reverse=True)
            for c in cands:
                jp = c / 'data.jsonl'
                if jp.exists():
                    print(f'[MMEDataset] Auto-resolved: {jp}')
                    return self._read_jsonl(jp, limit)
        if auto_download:
            print(f'[MMEDataset] No local data found; downloading from HuggingFace ({dataset_id})...')
            return self._download_and_cache(out_root=out_root, dataset_id=dataset_id, split=split, limit=limit, hf_endpoint=hf_endpoint)
        raise FileNotFoundError(f'[MMEDataset] Cannot find prepared  Run vmas_local_prepare.py first, or pass data_jsonl=... explicitly.')

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

    def _download_and_cache(self, out_root: str, dataset_id: str, split: str, limit: Optional[int], hf_endpoint: Optional[str]) -> List[Dict[str, Any]]:
        try:
            from skillgraph_datasets import load_dataset
            from PIL import Image as PILImage
            import jsonlines
        except ImportError as e:
            raise RuntimeError('Auto-download requires: pip install datasets pillow jsonlines') from e
        if hf_endpoint:
            os.environ['HF_ENDPOINT'] = hf_endpoint
        safe_id = self._safe_dataset_id(dataset_id)
        tag = f'{safe_id}__{split}' + (f'__limit{limit}' if limit else '')
        base = Path(out_root) / 'mme' / tag
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
                q = str(row.get('question') or row.get('text') or row.get('prompt') or '')
                ans = str(row.get('answer') or row.get('label') or row.get('gt_answer') or '').strip().lower()
                img_obj = row.get('image') or row.get('img') or row.get('image_path') or row.get('img_path')
                ex_id = str(row.get('id') or row.get('index') or i)
                img_path = images_dir / f'{ex_id}.png'
                _save_any_image(img_obj, img_path)
                rec = {'id': f'mme:{ex_id}', 'image': str(img_path.resolve()), 'question': q, 'choices': None, 'answer': ans, 'meta': {'source': 'MME', 'dataset_id': dataset_id, 'split': split}}
                w.write(rec)
                records.append(rec)
        return records

    @staticmethod
    def get_domain() -> str:
        return 'mme'

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
        Returns {"task": <question_text>, "image": <abs_image_path>}.
        The question text instructs the model to answer Yes or No.
        """
        q = record['question']
        task = f'{q}\n\nPlease answer with exactly one word: Yes or No.'
        return {'task': task, 'image': str(record.get('image') or '')}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """
        Normalise model output to 'yes' or 'no'.
        Mirrors MMLUDataset.postprocess_answer logic.
        """
        if isinstance(answer, list):
            answer = answer[0] if answer else ''
        if not isinstance(answer, str):
            raise Exception(f'Expected string, got {type(answer)}')
        t = answer.strip().lower()
        ans_pos = t.find('answer is')
        if ans_pos != -1:
            t = t[ans_pos + len('answer is'):].strip(':').strip()
        if t.startswith('yes'):
            return 'yes'
        if t.startswith('no'):
            return 'no'
        if 'yes' in t.split():
            return 'yes'
        if 'no' in t.split():
            return 'no'
        first = t.split()[0] if t.split() else t
        return first

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        ans = str(record.get('answer') or '').strip().lower()
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
