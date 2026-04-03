import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mra import (  # noqa: E402
    DetectionConfig,
    ExperimentConfig,
    LossConfig,
    TrainConfig,
    build_default_experiment,
    run_experiment,
)


def make_spec(
    experiment: ExperimentConfig,
    group: str,
    category: str,
    parameter_name: str,
    parameter_value,
) -> dict:
    return {
        "experiment": experiment,
        "group": group,
        "category": category,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
    }


def build_ablation_specs() -> list[dict]:
    base = build_default_experiment()
    ablation_root = "./test/checkpoints"

    specs = [
        make_spec(base, "ablation", "reference", "variant", "full"),
        make_spec(
            replace(
                base,
                name="no_graph",
                description="去掉自适应图学习，GCN 使用单位邻接",
                model=replace(base.model, use_graph=False),
                checkpoint_root=ablation_root,
                checkpoint_prefix="no_graph",
            ),
            "ablation",
            "module",
            "variant",
            "no_graph",
        ),
        make_spec(
            replace(
                base,
                name="no_gcn",
                description="时间分支去掉 GCN，仅保留 TCN",
                model=replace(base.model, use_gcn=False),
                checkpoint_root=ablation_root,
                checkpoint_prefix="no_gcn",
            ),
            "ablation",
            "module",
            "variant",
            "no_gcn",
        ),
        make_spec(
            replace(
                base,
                name="no_tcn",
                description="时间分支去掉 TCN，仅保留 GCN",
                model=replace(base.model, use_tcn=False),
                checkpoint_root=ablation_root,
                checkpoint_prefix="no_tcn",
            ),
            "ablation",
            "module",
            "variant",
            "no_tcn",
        ),
        make_spec(
            replace(
                base,
                name="no_freq",
                description="去掉频域补全分支，仅使用时间分支",
                model=replace(base.model, use_freq=False, fusion_mode="time"),
                checkpoint_root=ablation_root,
                checkpoint_prefix="no_freq",
            ),
            "ablation",
            "module",
            "variant",
            "no_freq",
        ),
        make_spec(
            replace(
                base,
                name="mean_fusion",
                description="将门控融合替换为简单平均融合",
                model=replace(base.model, fusion_mode="mean"),
                checkpoint_root=ablation_root,
                checkpoint_prefix="mean_fusion",
            ),
            "ablation",
            "module",
            "variant",
            "mean_fusion",
        ),
        make_spec(
            replace(
                base,
                name="no_sampling_embed",
                description="去掉采样率类型嵌入",
                model=replace(base.model, use_sampling_embedding=False),
                checkpoint_root=ablation_root,
                checkpoint_prefix="no_sampling_embed",
            ),
            "ablation",
            "module",
            "variant",
            "no_sampling_embed",
        ),
        make_spec(
            replace(
                base,
                name="recon_only_loss",
                description="损失函数仅保留重构项",
                loss=LossConfig(freq_weight=0.0, sparsity_weight=0.0),
                checkpoint_root=ablation_root,
                checkpoint_prefix="recon_only_loss",
            ),
            "ablation",
            "loss",
            "variant",
            "recon_only_loss",
        ),
    ]
    return specs


def build_sensitivity_specs() -> list[dict]:
    base = build_default_experiment()
    single_seed_train = replace(base.train, seeds=(40,))
    sensitivity_root = "./test/checkpoints"

    specs = [
        make_spec(
            replace(
                base,
                name="seq_len_30",
                description="窗口长度敏感性分析: seq_len=30",
                model=replace(base.model, seq_len=30),
                train=single_seed_train,
                checkpoint_root=sensitivity_root,
                checkpoint_prefix="seq_len_30",
            ),
            "sensitivity",
            "training",
            "seq_len",
            30,
        ),
        make_spec(
            replace(
                base,
                name="seq_len_50",
                description="窗口长度敏感性分析: seq_len=50",
                train=single_seed_train,
                checkpoint_root=base.checkpoint_root,
                checkpoint_prefix=base.checkpoint_prefix,
            ),
            "sensitivity",
            "training",
            "seq_len",
            50,
        ),
        make_spec(
            replace(
                base,
                name="seq_len_70",
                description="窗口长度敏感性分析: seq_len=70",
                model=replace(base.model, seq_len=70),
                train=single_seed_train,
                checkpoint_root=sensitivity_root,
                checkpoint_prefix="seq_len_70",
            ),
            "sensitivity",
            "training",
            "seq_len",
            70,
        ),
        make_spec(
            replace(
                base,
                name="d_model_32",
                description="隐空间维度敏感性分析: d_model=32",
                model=replace(base.model, d_model=32),
                train=single_seed_train,
                checkpoint_root=sensitivity_root,
                checkpoint_prefix="d_model_32",
            ),
            "sensitivity",
            "training",
            "d_model",
            32,
        ),
        make_spec(
            replace(
                base,
                name="d_model_64",
                description="隐空间维度敏感性分析: d_model=64",
                train=single_seed_train,
                checkpoint_root=base.checkpoint_root,
                checkpoint_prefix=base.checkpoint_prefix,
            ),
            "sensitivity",
            "training",
            "d_model",
            64,
        ),
        make_spec(
            replace(
                base,
                name="d_model_96",
                description="隐空间维度敏感性分析: d_model=96",
                model=replace(base.model, d_model=96),
                train=single_seed_train,
                checkpoint_root=sensitivity_root,
                checkpoint_prefix="d_model_96",
            ),
            "sensitivity",
            "training",
            "d_model",
            96,
        ),
        make_spec(
            replace(
                base,
                name="lr_5e-4",
                description="学习率敏感性分析: lr=5e-4",
                train=replace(single_seed_train, lr=5e-4),
                checkpoint_root=sensitivity_root,
                checkpoint_prefix="lr_5e-4",
            ),
            "sensitivity",
            "training",
            "lr",
            5e-4,
        ),
        make_spec(
            replace(
                base,
                name="lr_1e-3",
                description="学习率敏感性分析: lr=1e-3",
                train=single_seed_train,
                checkpoint_root=base.checkpoint_root,
                checkpoint_prefix=base.checkpoint_prefix,
            ),
            "sensitivity",
            "training",
            "lr",
            1e-3,
        ),
        make_spec(
            replace(
                base,
                name="lr_2e-3",
                description="学习率敏感性分析: lr=2e-3",
                train=replace(single_seed_train, lr=2e-3),
                checkpoint_root=sensitivity_root,
                checkpoint_prefix="lr_2e-3",
            ),
            "sensitivity",
            "training",
            "lr",
            2e-3,
        ),
        make_spec(
            replace(
                base,
                name="alpha_0.01",
                description="检测参数敏感性分析: alpha=0.01",
                detection=DetectionConfig(ewma_alpha=0.01, threshold_std=1.5, min_run=150),
            ),
            "sensitivity",
            "detection",
            "ewma_alpha",
            0.01,
        ),
        make_spec(
            replace(
                base,
                name="alpha_0.02",
                description="检测参数敏感性分析: alpha=0.02",
                detection=DetectionConfig(ewma_alpha=0.02, threshold_std=1.5, min_run=150),
            ),
            "sensitivity",
            "detection",
            "ewma_alpha",
            0.02,
        ),
        make_spec(
            replace(
                base,
                name="alpha_0.05",
                description="检测参数敏感性分析: alpha=0.05",
                detection=DetectionConfig(ewma_alpha=0.05, threshold_std=1.5, min_run=150),
            ),
            "sensitivity",
            "detection",
            "ewma_alpha",
            0.05,
        ),
        make_spec(
            replace(
                base,
                name="threshold_std_1.0",
                description="检测参数敏感性分析: threshold_std=1.0",
                detection=DetectionConfig(ewma_alpha=0.02, threshold_std=1.0, min_run=150),
            ),
            "sensitivity",
            "detection",
            "threshold_std",
            1.0,
        ),
        make_spec(
            replace(
                base,
                name="threshold_std_1.5",
                description="检测参数敏感性分析: threshold_std=1.5",
                detection=DetectionConfig(ewma_alpha=0.02, threshold_std=1.5, min_run=150),
            ),
            "sensitivity",
            "detection",
            "threshold_std",
            1.5,
        ),
        make_spec(
            replace(
                base,
                name="threshold_std_2.0",
                description="检测参数敏感性分析: threshold_std=2.0",
                detection=DetectionConfig(ewma_alpha=0.02, threshold_std=2.0, min_run=150),
            ),
            "sensitivity",
            "detection",
            "threshold_std",
            2.0,
        ),
        make_spec(
            replace(
                base,
                name="min_run_100",
                description="检测参数敏感性分析: min_run=100",
                detection=DetectionConfig(ewma_alpha=0.02, threshold_std=1.5, min_run=100),
            ),
            "sensitivity",
            "detection",
            "min_run",
            100,
        ),
        make_spec(
            replace(
                base,
                name="min_run_150",
                description="检测参数敏感性分析: min_run=150",
                detection=DetectionConfig(ewma_alpha=0.02, threshold_std=1.5, min_run=150),
            ),
            "sensitivity",
            "detection",
            "min_run",
            150,
        ),
        make_spec(
            replace(
                base,
                name="min_run_200",
                description="检测参数敏感性分析: min_run=200",
                detection=DetectionConfig(ewma_alpha=0.02, threshold_std=1.5, min_run=200),
            ),
            "sensitivity",
            "detection",
            "min_run",
            200,
        ),
    ]
    return specs


def build_specs(group: str) -> list[dict]:
    if group == "ablation":
        return build_ablation_specs()
    if group == "sensitivity":
        return build_sensitivity_specs()
    if group == "all":
        return build_ablation_specs() + build_sensitivity_specs()
    raise ValueError(f"Unsupported group: {group}")


def save_summary(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    if "full" in df["experiment"].values:
        full_row = df[df["experiment"] == "full"].iloc[0]
        for metric in ["persistent_f1", "persistent_recall", "persistent_precision", "raw_f1"]:
            df[f"delta_{metric}"] = df[metric] - float(full_row[metric])

    df.to_csv(output_root / "all_metrics.csv", index=False)

    ablation_df = df[df["group"] == "ablation"].copy()
    if not ablation_df.empty:
        ablation_df.to_csv(output_root / "ablation_metrics.csv", index=False)

    sensitivity_df = df[df["group"] == "sensitivity"].copy()
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(output_root / "sensitivity_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation and sensitivity experiments for AGF-ADNet.")
    parser.add_argument("--group", default="all", choices=["ablation", "sensitivity", "all"])
    parser.add_argument("--only", nargs="*", default=None, help="Run only the specified experiment names.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached checkpoints and retrain.")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_specs(args.group)

    if args.only:
        only_set = set(args.only)
        specs = [spec for spec in specs if spec["experiment"].name in only_set]

    if args.list:
        for spec in specs:
            print(f"{spec['experiment'].name}\t{spec['group']}\t{spec['parameter_name']}={spec['parameter_value']}")
        return

    rows = []
    for spec in specs:
        result = run_experiment(
            spec["experiment"],
            device=args.device,
            force_retrain=args.force_retrain,
        )
        row = {
            "experiment": spec["experiment"].name,
            "group": spec["group"],
            "category": spec["category"],
            "parameter_name": spec["parameter_name"],
            "parameter_value": spec["parameter_value"],
            "output_dir": result["output_dir"],
            **result["metrics"],
        }
        rows.append(row)
        save_summary(rows, ROOT / "test" / "results")

    save_summary(rows, ROOT / "test" / "results")


if __name__ == "__main__":
    main()
