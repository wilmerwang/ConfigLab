#!/usr/bin/env python3
"""results_helper.py - 完整版 ColumnConfig 支持.

功能:
- 聚合 train/test/predict run
- train val/aupr 最优摘要, 性能指标高亮
- commit 前7位
- epoch/step int, 性能指标保留4位小数
- inspect: .hydra/config.yaml + log + checkpoint
- compare: 多 run config 差异 + checkpoint, 带ID
- YAML 美化
- 自定义 ColumnConfig 列表支持不同任务显示
- 修复无 CSV run 导致 ID 对不上问题
"""

import json
from dataclasses import dataclass
from pathlib import Path
from shutil import move

import pandas as pd
import yaml
from rich import box
from rich.console import Console
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

console = Console()


# =========================
# Column 配置
# =========================
@dataclass
class ColumnConfig:
    name: str
    display_name: str | None = None
    mode: str | None = None  # "max", "min", None


# 默认/任务列配置
default_columns = None  # None 表示提取 CSV 所有列

train_columns = [
    ColumnConfig("epoch", "Epoch"),
    ColumnConfig("val/acc", "Val Acc", "max"),
    ColumnConfig("val/aupr", "Val AUPR", "max"),
    ColumnConfig("val/auroc", "Val AUROC", "max"),
    ColumnConfig("val/f1", "Val F1", "max"),
    ColumnConfig("val/loss", "Val Loss", "min"),
]

test_columns = [
    ColumnConfig("epoch", "Epoch"),
    ColumnConfig("test/acc", "Test Acc", "max"),
    ColumnConfig("test/aupr", "Test AUPR", "max"),
    ColumnConfig("test/auroc", "Test AUROC", "max"),
    ColumnConfig("test/f1", "Test F1", "max"),
]


# =========================
# 工具函数
# =========================
def read_git_commit(run_dir: str) -> str:
    git_file = Path(run_dir) / "git_snapshot/git_snapshot.json"
    if git_file.exists():
        try:
            data = json.load(open(git_file))
            commit = data.get("commit", "")
            return commit[:7]
        except Exception:
            return ""
    return ""


def get_best_val_aupr(metrics_csv: Path, column_configs: list[ColumnConfig] | None = None) -> dict | None:
    df = pd.read_csv(metrics_csv)
    column_names = df.columns.tolist() if column_configs is None else [c.name for c in column_configs]

    # 只保留存在的列
    column_names = [c for c in column_names if c in df.columns]
    if "val/aupr" not in column_names:
        return None

    df_valid = df.dropna(subset=["val/aupr"])
    if df_valid.empty:
        return None
    best_idx = df_valid["val/aupr"].idxmax()
    best_row = df_valid.loc[best_idx]
    summary = best_row[column_names].to_dict()

    # 格式化
    for k, v in summary.items():
        if k in {"epoch", "step"}:
            summary[k] = int(v)
        elif isinstance(v, (float, int)):
            summary[k] = round(float(v), 4)
    return summary


def clean_failed_runs(task_dir: str, dry_run: bool = True) -> None:
    """清理失败实验 (没有 metrics.csv 或没有 output 开头文件) dry_run=True 时只打印, 不执行移动."""
    task_dir = Path(task_dir)
    failed_dir = task_dir / "failed_runs"
    failed_dir.mkdir(exist_ok=True)

    runs_dir = task_dir / "runs"
    if not runs_dir.exists():
        console.print(f"[red]任务目录不存在: {runs_dir}[/red]")
        return

    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue

        # 检查是否有 metrics.csv
        metrics = next(run.rglob("metrics.csv"), None)
        # 检查是否有 output* 文件
        output_files = list(run.glob("output*"))

        if metrics is None and not output_files:
            console.print(f"[red]失败 run: {run.name}[/red]")
            if dry_run:
                console.print("    [yellow]Dry run: 不移动[/yellow]")
            else:
                move(str(run), failed_dir / run.name)
                console.print(f"    [green]已移动到 {failed_dir}[/green]")


def read_metrics(run_dir: str, task: str, column_configs: list[ColumnConfig] | None = None) -> dict | None:
    run_dir = Path(run_dir)
    metrics_file = next(run_dir.rglob("metrics.csv"), None)
    if metrics_file is None:
        return None
    if task == "train":
        return get_best_val_aupr(metrics_file, column_configs)
    df = pd.read_csv(metrics_file)
    if df.empty:
        return None
    if column_configs is None:
        col_names = df.columns.tolist()
    else:
        col_names = [c.name for c in column_configs if c.name in df.columns]
    row = df.iloc[0][col_names].to_dict()
    for k, v in row.items():
        if k in {"epoch", "step"}:
            row[k] = int(v)
        elif isinstance(v, (float, int)):
            row[k] = round(float(v), 4)
    return row


# =========================
# 聚合模块
# =========================
def aggregate_task(task_dir: str, task: str, column_configs: list[ColumnConfig] | None = None) -> list[dict]:
    task_dir = Path(task_dir)
    runs_dir = task_dir / "runs"
    if not runs_dir.exists():
        console.print(f"[red]任务目录不存在: {runs_dir}[/red]")
        return []

    aggregated = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        metrics = read_metrics(run, task, column_configs)
        if metrics is None:
            continue
        metrics["run_dir"] = str(run)
        metrics["commit"] = read_git_commit(run)
        aggregated.append(metrics)

    # 给聚合后的 run 连续 ID
    for idx, run in enumerate(aggregated):
        run["ID"] = idx
    return aggregated


# =========================
# 显示模块
# =========================
def display_summary(aggregated: list[dict], column_configs: list[ColumnConfig] | None = None) -> None:
    if not aggregated:
        console.print("[yellow]没有找到任何有效 run[/yellow]")
        return

    if column_configs is None:
        metrics_cols = [k for k in aggregated[0] if k not in {"run_dir", "commit", "ID"}]
        column_configs = [ColumnConfig(name=c) for c in metrics_cols]

    max_vals = {}
    min_vals = {}
    for col_cfg in column_configs:
        vals = [r[col_cfg.name] for r in aggregated if isinstance(r[col_cfg.name], (float, int))]
        max_vals[col_cfg.name] = max(vals) if vals else None
        min_vals[col_cfg.name] = min(vals) if vals else None

    table = Table(title="实验最优摘要", box=box.SIMPLE_HEAVY)
    table.add_column("ID", justify="right")
    table.add_column("run_dir")
    table.add_column("commit")
    for col_cfg in column_configs:
        table.add_column(col_cfg.display_name or col_cfg.name)

    for run in aggregated:
        row = [str(run["ID"]), run["run_dir"], run.get("commit", "")]
        for col_cfg in column_configs:
            v = run.get(col_cfg.name)
            if isinstance(v, (float, int)):
                if (col_cfg.mode == "max" and v == max_vals[col_cfg.name]) or (
                    col_cfg.mode == "min" and v == min_vals[col_cfg.name]
                ):
                    v_str = f"[bold red]{v}[/bold red]"
                else:
                    v_str = str(v)
            else:
                v_str = str(v)
            row.append(v_str)
        table.add_row(*row)

    console.print(table)


# =========================
# inspect 模块
# =========================
def print_yaml_file(file_path: Path) -> None:
    text = file_path.read_text()
    syntax = Syntax(text, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)


def inspect_run(run_dir: str) -> None:
    run_dir = Path(run_dir)
    console.print(f"[bold green]Inspect {run_dir}[/bold green]")

    config_file = run_dir / ".hydra/config.yaml"
    if config_file.exists():
        print_yaml_file(config_file)
    else:
        console.print("[red]No config found[/red]")

    for log_file in run_dir.glob("*.log"):
        console.print(f"[blue]{log_file.name}[/blue]")
        console.print(log_file.read_text()[:1000] + "\n...")

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = [str(p) for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
        console.print(f"[magenta]Checkpoints:[/magenta] {ckpts if ckpts else 'None'}")


# =========================
# compare 多 run 模块
# =========================
def collect_key_paths(d: dict, prefix: list | None = None) -> list:
    if prefix is None:
        prefix = []
    paths = []
    for k, v in d.items():
        current = [*prefix, k]
        if isinstance(v, dict):
            paths.extend(collect_key_paths(v, current))
        else:
            paths.append(current)
    return paths


def values_differ(values: list) -> bool:
    first = values[0]
    return any(v != first for v in values[1:])


def get_differing_keys(dicts: list[dict]) -> dict:
    all_paths = set()
    for d in dicts:
        all_paths.update(tuple(p) for p in collect_key_paths(d))
    diffs = {}
    for path in all_paths:
        values = []
        for d in dicts:
            v = d
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            values.append(v)
        if values_differ(values):
            diffs[".".join(path)] = values
    return diffs


def compare_runs_multi(aggregated: list[dict], run_ids: list[int]) -> None:
    run_dirs = [aggregated[i]["run_dir"] for i in run_ids]
    ids = [aggregated[i]["ID"] for i in run_ids]
    run_names = [f"{id_}: {Path(d).name}" for id_, d in zip(ids, run_dirs, strict=False)]
    configs = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        cfg_file = run_dir / ".hydra/config.yaml"
        if cfg_file.exists():
            try:
                cfg = yaml.safe_load(cfg_file.read_text())
            except Exception:
                cfg = {}
        else:
            cfg = {}
        configs.append(cfg)

    diffs = get_differing_keys(configs)
    if not diffs:
        console.print("[green]所有 run 配置完全相同[/green]")
        return

    tree = Tree("Config Differences (only differing fields)")
    for k, vals in diffs.items():
        branch = tree.add(f"[bold]{k}[/bold]")
        for run_name, val in zip(run_names, vals, strict=False):
            val_str = yaml.dump(val, default_flow_style=False).strip() if isinstance(val, dict) else str(val)
            branch.add(f"[cyan]{run_name}[/cyan]: {val_str}")
    console.print(tree)

    console.print("\n[magenta]Checkpoints (non-last.ckpt)[/magenta]")
    for run_id, run_dir in zip(ids, run_dirs, strict=False):
        run_path = Path(run_dir)
        ckpt_dir = run_path / "checkpoints"
        ckpts = []
        if ckpt_dir.exists():
            ckpts = [str(p) for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
        console.print(f"[green]{run_id}: {run_path.name}[/green]: {ckpts if ckpts else 'None'}")


# =========================
# 交互模块
# =========================
def interactive_select(aggregated: list[dict], column_configs: list[ColumnConfig] | None = None) -> None:
    if not aggregated:
        return
    while True:
        choice = Prompt.ask("输入实验 ID 查看配置/对比, 多个 ID 用逗号, 输入 q 退出", default="")
        if choice.lower() in {"q", "quit", "exit"}:
            break
        ori_ids = [x.strip() for x in choice.split(",") if x.strip().isdigit()]
        ids = [int(x) for x in ori_ids if 0 <= int(x) < len(aggregated)]
        if len(ori_ids) != len(ids):
            console.print("[red]输入中包含无效 ID[/red]")
            continue
        if len(ids) == 1:
            inspect_run(aggregated[ids[0]]["run_dir"])
        else:
            compare_runs_multi(aggregated, ids)


# =========================
# CLI 主入口
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验结果辅助工具")
    parser.add_argument("--task_dir", type=str, required=True, help="任务目录, 例如 train/test/predict")
    parser.add_argument("--task", type=str, choices=["train", "test", "predict"], help="任务类型")
    parser.add_argument("--clean_failed", action="store_true", help="清理失败实验")
    args = parser.parse_args()

    if not args.task_dir:
        console.print("[red]请提供有效的任务目录[/red]")
        exit(1)

    if args.clean_failed:
        clean_failed_runs(args.task_dir, dry_run=False)

    if not args.task:
        if not args.clean_failed:
            console.print("[red]请提供有效的任务类型或者清理失败实验[/red]")
            exit(1)
        exit(0)

    # 根据任务选择列配置
    if args.task == "train":
        columns = train_columns
    elif args.task == "test":
        columns = test_columns
    else:
        columns = default_columns

    aggregated = aggregate_task(args.task_dir, args.task, column_configs=columns)
    display_summary(aggregated, column_configs=columns)
    interactive_select(aggregated, column_configs=columns)
