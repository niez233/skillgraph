"""
用法：
    python experiments/analyze_results.py result/mmbench_FullConnected_xxx.jsonl
    python experiments/analyze_results.py result/mmbench_FullConnected_xxx.jsonl --show_errors
    python experiments/analyze_results.py result/file1.jsonl result/file2.jsonl  # 对比两次实验
"""
import json
import sys
import argparse
from pathlib import Path
from collections import Counter


def load_jsonl(path: str):
    summary = None
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get('_type') == 'summary':
                summary = obj
            else:
                records.append(obj)
    return summary, records


def analyze(path: str, show_errors: bool = False):
    summary, records = load_jsonl(path)
    print(f"\n{'='*60}")
    print(f"文件: {path}")

    if summary:
        print(f"总准确率 : {summary['score']:.1%}  ({summary['correct']}/{summary['total']})")
        print(f"模式     : {summary['mode']} {summary['timestamp']}")

    errors = [r for r in records if not r['is_correct']]
    corrects = [r for r in records if r['is_correct']]
    print(f"\n正确: {len(corrects)}  错误: {len(errors)}  共: {len(records)}")

    # 错误分布：模型预测了什么
    pred_counter = Counter(r['predicted'] for r in errors)
    print(f"\n错题中模型的预测分布（预测→次数）:")
    for letter, cnt in sorted(pred_counter.items(), key=lambda x: -x[1]):
        print(f"  预测 {letter}: {cnt} 次")

    # 错误分布：正确答案是什么（哪个选项最难）
    gt_counter = Counter(r['correct'] for r in errors)
    print(f"\n错题中正确答案分布（正确答案→次数）:")
    for letter, cnt in sorted(gt_counter.items(), key=lambda x: -x[1]):
        print(f"  正确答案 {letter}: {cnt} 次")

    # 混淆矩阵
    print(f"\n混淆矩阵 (行=正确答案, 列=模型预测):")
    letters = sorted(set(r['correct'] for r in records) | set(r['predicted'] for r in records))
    header = '     ' + '  '.join(f'{l:>3}' for l in letters)
    print(header)
    for gt in letters:
        row_records = [r for r in records if r['correct'] == gt]
        pred_dist = Counter(r['predicted'] for r in row_records)
        row = f"  {gt}  " + '  '.join(f"{pred_dist.get(p, 0):>3}" for p in letters)
        print(row)

    # 打印错题详情
    if show_errors:
        print(f"\n{'='*60}")
        print(f"错题列表（共 {len(errors)} 题）:")
        for i, r in enumerate(errors, 1):
            print(f"\n[{i}] id={r['id']}")
            print(f"  问题  : {r['question'][:80]}{'...' if len(r['question'])>80 else ''}")
            print(f"  预测  : {r['predicted']}  正确: {r['correct']}")


def compare(paths: list):
    """对比多个实验文件的总分"""
    print(f"\n{'='*60}")
    print("实验对比:")
    print(f"{'文件':<50} {'准确率':>8} {'正确/总数':>12}")
    print('-' * 72)
    for path in paths:
        summary, records = load_jsonl(path)
        name = Path(path).name
        if summary:
            print(f"{name:<50} {summary['score']:>8.1%} {summary['correct']:>6}/{summary['total']:<6}")
        else:
            correct = sum(1 for r in records if r['is_correct'])
            total = len(records)
            score = correct / total if total else 0
            print(f"{name:<50} {score:>8.1%} {correct:>6}/{total:<6}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='一个或多个 .jsonl 结果文件')
    parser.add_argument('--show_errors', action='store_true', help='打印所有错题详情')
    args = parser.parse_args()

    if len(args.files) == 1:
        analyze(args.files[0], show_errors=args.show_errors)
    else:
        compare(args.files)
        # 如果同时想看错题，对每个文件单独分析
        if args.show_errors:
            for f in args.files:
                analyze(f, show_errors=True)
