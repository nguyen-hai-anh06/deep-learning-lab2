"""Collect validation results from all teammates without touching the test set."""

import argparse
import csv
import glob
import json
import os

from tabulate import tabulate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(args.experiments_root, "validation_summary")
    rows = []
    for summary_path in glob.glob(
        os.path.join(args.experiments_root, "**", "summary.json"), recursive=True
    ):
        run_dir = os.path.dirname(summary_path)
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)
        if summary.get("status") != "completed":
            continue
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as file:
            config = json.load(file)
        with open(os.path.join(run_dir, "split.json"), encoding="utf-8") as file:
            split = json.load(file)
        with open(os.path.join(run_dir, "history.json"), encoding="utf-8") as file:
            history = json.load(file)
        with open(os.path.join(run_dir, "environment.json"), encoding="utf-8") as file:
            environment = json.load(file)
        rows.append({
            "run_id": summary["run_id"],
            "member_id": summary["member_id"],
            "model": summary["model"],
            "strategy": summary["strategy"],
            "best_validation_accuracy": summary["best_validation_accuracy"],
            "best_epoch": summary["best_epoch"],
            "final_train_accuracy": history[-1]["train_accuracy"],
            "final_validation_accuracy": history[-1]["validation_accuracy"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "optimizer": config["optimizer"],
            "weight_decay": config["weight_decay"],
            "training_seconds": sum(item["epoch_seconds"] for item in history),
            "split_checksum": split["validation_indices_checksum"],
            "split_seed": split["seed"],
            "train_size": split["train_size"],
            "validation_size": split["validation_size"],
            "test_size": split["test_size"],
            "gpu": environment.get("gpu"),
            "source_fingerprint": environment.get("source_fingerprint"),
        })
    if not rows:
        raise FileNotFoundError("Chưa có run hoàn chỉnh")
    signatures = {
        (r["split_checksum"], r["split_seed"], r["train_size"], r["validation_size"], r["test_size"])
        for r in rows
    }
    if len(signatures) != 1:
        raise RuntimeError("Các thành viên dùng data split khác nhau")
    fingerprints = {r["source_fingerprint"] for r in rows if r["source_fingerprint"]}
    if len(fingerprints) != 1:
        raise RuntimeError("Các thành viên không dùng cùng phiên bản source code")
    best_by_model = {}
    for row in rows:
        previous = best_by_model.get(row["model"])
        if previous is None or row["best_validation_accuracy"] > previous["best_validation_accuracy"]:
            best_by_model[row["model"]] = row
    for row in rows:
        row["selected_for_test"] = row is best_by_model[row["model"]]
    rows.sort(key=lambda item: (item["model"], -item["best_validation_accuracy"]))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "validation_results.json"), "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "validation_results.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    table = [[r["model"], r["strategy"], r["member_id"], f"{r['best_validation_accuracy']:.2f}", "yes" if r["selected_for_test"] else ""] for r in rows]
    print(tabulate(table, headers=["Model", "Strategy", "Member", "Best val %", "Selected"], tablefmt="github"))


if __name__ == "__main__":
    main()
