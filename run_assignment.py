"""Run a teammate's experiment queue, resuming interrupted Colab runs automatically."""

import argparse
import json
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, help="A persistent Google Drive directory")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    with open(config_path, encoding="utf-8") as file:
        assignment = json.load(file)

    member_id = assignment["member_id"]
    for experiment in assignment["experiments"]:
        run_id = experiment["run_id"]
        run_dir = os.path.join(args.output_root, member_id, run_id)
        summary_path = os.path.join(run_dir, "summary.json")
        last_path = os.path.join(run_dir, "checkpoints", "last.pt")
        if os.path.exists(summary_path) and not args.rerun_completed:
            saved_config_path = os.path.join(run_dir, "config.json")
            with open(saved_config_path, encoding="utf-8") as file:
                saved_config = json.load(file)
            expected = {
                "model": experiment["model"],
                "strategy": experiment["strategy"],
                "epochs": experiment["epochs"],
                "batch_size": experiment["batch_size"],
                "learning_rate": experiment["learning_rate"],
                "optimizer": experiment["optimizer"],
                "weight_decay": experiment["weight_decay"],
                "seed": assignment.get("seed", 42),
            }
            changed = [key for key, value in expected.items() if saved_config.get(key) != value]
            if changed:
                raise ValueError(
                    f"Run {run_id} đã tồn tại nhưng cấu hình khác ở {changed}. "
                    "Hãy dùng run_id mới để giữ nguyên kết quả cũ."
                )
            print(f"[Skip completed] {run_id}")
            continue
        command = [
            sys.executable,
            "-u",
            os.path.join(script_dir, "train.py"),
            "--model", experiment["model"],
            "--strategy", experiment["strategy"],
            "--epochs", str(experiment["epochs"]),
            "--batch-size", str(experiment["batch_size"]),
            "--lr", str(experiment["learning_rate"]),
            "--optimizer", experiment["optimizer"],
            "--weight-decay", str(experiment["weight_decay"]),
            "--seed", str(assignment.get("seed", 42)),
            "--member-id", member_id,
            "--run-id", run_id,
            "--output-root", args.output_root,
            "--data-dir", args.data_dir,
        ]
        if args.subset:
            command.extend(["--subset", str(args.subset)])
        if os.path.exists(last_path):
            command.extend(["--resume", "auto"])
        print(f"[Run] {' '.join(command)}")
        os.makedirs(run_dir, exist_ok=True)
        console_path = os.path.join(run_dir, "console.log")
        with open(console_path, "a", encoding="utf-8") as console_file:
            process = subprocess.Popen(
                command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                console_file.write(line)
                console_file.flush()
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


if __name__ == "__main__":
    main()
