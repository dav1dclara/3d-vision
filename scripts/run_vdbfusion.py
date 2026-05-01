import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def main() ->None:
    parser = argparse.ArgumentParser(
        description="VDBFusion reconstruction (supported public workflow)"
    )
    parser.add_argument(
        "--config",
        default="configs/vdbfusion_config.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    # vdbfusion expects to be run from repo root
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "vdbfusion_reconstruction.py"

    if not script_path.exists():
        raise FileNotFoundError(script_path)

    cmd = [
        "python",
        str(script_path),
        "--config",
        str(config_path),
    ]

    print("Running VDBFusion pipeline:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=repo_root, check=True)

    print(f"Done. Total time: {datetime.now() - start}")


if __name__ == "__main__":
    main()
