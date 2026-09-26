"""Run the final two-scene acceptance suite with real local models."""

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.acceptance import build_default_scenarios, run_acceptance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重复验证边防、火灾离线 Demo")
    parser.add_argument("--repeat", type=int, default=3, help="每个场景重复次数")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="本轮验收输出目录；默认写入 runs/acceptance/<时间>",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "backup",
        help="保存每个场景最后一次通过结果的目录",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = args.output_root or PROJECT_ROOT / "runs" / "acceptance" / run_name
    try:
        report = run_acceptance(
            build_default_scenarios(PROJECT_ROOT),
            repeat=args.repeat,
            output_root=output_root,
            backup_root=args.backup_root,
        )
    except Exception as error:
        print(f"验收失败：{error}", file=sys.stderr)
        return 1
    print(f"验收通过：{report['run_count']} 次运行")
    print(f"报告：{output_root / 'acceptance_report.json'}")
    print(f"备用结果：{report['backup_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
