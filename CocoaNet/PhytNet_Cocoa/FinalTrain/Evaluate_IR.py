from __future__ import print_function
from __future__ import division

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn as nn
from sklearn import metrics
from torchvision import datasets, transforms


parser = argparse.ArgumentParser("Evaluate trained model(s)")
parser.add_argument("--model_name", type=str, default=None,
                    help="Single-model mode: model name used in <root>/models/<model_name>.pth")
parser.add_argument("--model_path", type=str, default=None,
                    help="Single-model mode: explicit .pth path (overrides --model_name)")
parser.add_argument("--root", type=str, default="/users/jrs596/scratch",
                    help="Root directory")
parser.add_argument("--data_dir", type=str, default=None,
                    help="Single-model mode dataset directory relative to root")
parser.add_argument("--test_set", type=str, default="test",
                    help="Single-model mode split name inside dataset (e.g. test or val)")
parser.add_argument("--test_dir", type=str,
                    default="/users/jrs596/scratch/HiveSight-Dataset-V2/export_split_rgb500_dynamic_single/test",
                    help="Batch mode test directory")
parser.add_argument("--test_dir_notcocoa", type=str,
                    default="/users/jrs596/scratch/HiveSight-Dataset-V2/export_split_rgb500_dynamic_single/test_NotCocoa",
                    help="Batch mode test directory for model names containing NotCocoa")
parser.add_argument("--arch", type=str, default="PhytNetV0",
                    help="Fallback architecture name")
parser.add_argument("--input_size", type=int, default=None,
                    help="Optional resize applied before evaluation")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for evaluation")
parser.add_argument("--GPU", type=str, default="0",
                    help="GPU id")
parser.add_argument("--batch_redo", action="store_true", default=False,
                    help="Evaluate all models in --models_dir ending with --model_suffix")
parser.add_argument("--models_dir", type=str, default="/users/jrs596/scratch/models",
                    help="Directory containing model .pth files")
parser.add_argument("--model_suffix", type=str, default="redo.pth",
                    help="Suffix filter for batch mode")
parser.add_argument("--output_format", type=str, choices=["csv", "json"], default="csv",
                    help="Output summary format")
parser.add_argument("--output_path", type=str, default=None,
                    help="Output file path. Default: <models_dir>/eval_redo_results.<ext>")
args = parser.parse_args()


script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "utils"))
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

import toolbox
from training_loop import _extract_logits


def infer_arch_from_state(state, fallback_arch):
    keys = set(state.keys())
    if any(k.startswith("layer1.0.residual.") for k in keys):
        return "PhytNetV0"
    if "classifier.2.weight" in keys:
        return "convnext_tiny"
    if any(k.startswith("layer1.0.conv3.") for k in keys):
        return "resnet50"
    if any(k.startswith("layer1.0.conv1.") for k in keys):
        return "resnet18"
    return fallback_arch


def get_default_config(arch, num_classes, model_name=None):
    if arch == "PhytNetV0":
        model_name = (model_name or "").lower()
        if "183" in model_name:
            return {
                "beta1": 0.9671538235629524,
                "beta2": 0.9574398373980104,
                "dim_1": 126,
                "dim_2": 91,
                "dim_3": 89,
                "input_size": 371,
                "kernel_1": 5,
                "kernel_2": 1,
                "kernel_3": 17,
                "learning_rate": 9.66816458944127e-05,
                "num_blocks_1": 2,
                "num_blocks_2": 1,
                "out_channels": num_classes,
            }
        if "332" in model_name:
            return {
                "beta1": 0.9657828624377116,
                "beta2": 0.9908102731106424,
                "dim_1": 104,
                "dim_2": 109,
                "dim_3": 110,
                "input_size": 350,
                "kernel_1": 5,
                "kernel_2": 7,
                "kernel_3": 13,
                "learning_rate": 0.00013365304940966892,
                "num_blocks_1": 1,
                "num_blocks_2": 2,
                "out_channels": num_classes,
            }
        return {
            "beta1": 0.9650025364732508,
            "beta2": 0.981605256508036,
            "dim_1": 79,
            "dim_2": 107,
            "dim_3": 93,
            "input_size": 415,
            "kernel_1": 5,
            "kernel_2": 1,
            "kernel_3": 7,
            "learning_rate": 0.0002975957026209971,
            "num_blocks_1": 2,
            "num_blocks_2": 1,
            "out_channels": num_classes,
        }
    if arch in ("resnet18", "resnet50"):
        return {
            "learning_rate": 5.140551979453639e-05,
            "input_size": 375,
            "beta1": 0.9841235203771193,
            "beta2": 0.9895409209654844,
            "out_channels": num_classes,
        }
    return {"out_channels": num_classes, "input_size": args.input_size}


def load_state(path, device):
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError("Model file does not contain a state_dict dict.")
    fixed = {}
    for k, v in state.items():
        nk = k.replace("module.", "").replace("_orig_mod.", "")
        fixed[nk] = v
    return fixed


def infer_num_classes_from_state(state):
    for key in ("fc.weight", "classifier.2.weight", "classifier.weight"):
        if key in state and hasattr(state[key], "shape") and len(state[key].shape) == 2:
            return int(state[key].shape[0])
    return None


def build_eval_loader(eval_dir, input_size, batch_size):
    if input_size is None:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])

    dataset = datasets.ImageFolder(eval_dir, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=6, worker_init_fn=toolbox.worker_init_fn
    )
    return dataset, loader


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = _extract_logits(model(inputs))
            loss = criterion(logits, labels)

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_count += batch_n

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total_count, 1)
    report = metrics.classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    return avg_loss, accuracy, precision, recall, f1, report, total_count


def evaluate_one(model_path, eval_dir, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    state = load_state(model_path, device)
    arch = infer_arch_from_state(state, args.arch)
    inferred_classes = infer_num_classes_from_state(state)

    tmp_dataset, _ = build_eval_loader(eval_dir, args.input_size, args.batch_size)
    dataset_classes = len(tmp_dataset.classes)
    num_classes = inferred_classes if inferred_classes is not None else dataset_classes

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    config = get_default_config(arch, num_classes, model_name=model_name)
    if args.input_size is not None:
        config["input_size"] = args.input_size

    model = toolbox.build_model(arch=arch, num_classes=num_classes, config=config).to(device)
    model.load_state_dict(state, strict=True)

    input_size = config.get("input_size", args.input_size)
    dataset, loader = build_eval_loader(eval_dir, input_size, args.batch_size)
    loss, acc, prec, rec, f1, report, total_n = evaluate(model, loader, device)

    per_class_f1 = {}
    for idx, class_name in enumerate(dataset.classes):
        k = str(idx)
        per_class_f1[class_name] = report[k]["f1-score"] if k in report else 0.0

    return {
        "model_name": os.path.splitext(os.path.basename(model_path))[0],
        "model_path": model_path,
        "arch": arch,
        "eval_dir": eval_dir,
        "num_samples": total_n,
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_class_f1": per_class_f1,
        "status": "ok",
    }


def resolve_single_eval_dir():
    if args.data_dir is None:
        raise ValueError("--data_dir is required for single-model mode")
    candidate = os.path.join(args.root, args.data_dir, args.test_set)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(args.test_set):
        return args.test_set
    raise FileNotFoundError(f"Could not find test directory: {candidate}")


def resolve_single_model_path():
    if args.model_path:
        return args.model_path
    if args.model_name:
        return os.path.join(args.root, "models", args.model_name + ".pth")
    raise ValueError("Provide --model_path or --model_name for single-model mode")


def resolve_output_path():
    if args.output_path:
        return args.output_path
    ext = "csv" if args.output_format == "csv" else "json"
    return os.path.join(args.models_dir, f"eval_redo_results.{ext}")


def write_results(results, output_path):
    if args.output_format == "json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return

    rows = []
    fieldnames = set()
    for r in results:
        row = {k: v for k, v in r.items() if k != "per_class_f1"}
        for cls, val in r.get("per_class_f1", {}).items():
            row[f"per_class_f1.{cls}"] = val
        rows.append(row)
        fieldnames.update(row.keys())

    preferred = [
        "model_name", "model_path", "arch", "eval_dir", "status", "error",
        "num_samples", "loss", "accuracy", "precision", "recall", "f1",
    ]
    ordered = [k for k in preferred if k in fieldnames] + sorted(k for k in fieldnames if k not in preferred)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_single(device):
    model_path = resolve_single_model_path()
    eval_dir = resolve_single_eval_dir()
    result = evaluate_one(model_path, eval_dir, device)

    print("\nEvaluation Results")
    print("=" * 40)
    print("Model path:", result["model_path"])
    print("Eval dir:", result["eval_dir"])
    print("Arch:", result["arch"])
    print("-" * 40)
    print(f"Loss      : {result['loss']:.6f}")
    print(f"Accuracy  : {result['accuracy']:.6f}")
    print(f"Precision : {result['precision']:.6f}")
    print(f"Recall    : {result['recall']:.6f}")
    print(f"F1        : {result['f1']:.6f}")
    print("-" * 40)
    print("Per-class F1")
    for cls, val in sorted(result["per_class_f1"].items()):
        print(f"{cls:>20s}: {val:.6f}")


def run_batch_redo(device):
    model_paths = sorted(
        os.path.join(args.models_dir, f)
        for f in os.listdir(args.models_dir)
        if f.endswith(args.model_suffix)
    )
    if not model_paths:
        raise FileNotFoundError(f"No models ending with '{args.model_suffix}' in {args.models_dir}")

    results = []
    print(f"Found {len(model_paths)} model(s) to evaluate")
    for i, model_path in enumerate(model_paths, 1):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        eval_dir = args.test_dir_notcocoa if "NotCocoa" in model_name else args.test_dir
        print(f"\n[{i}/{len(model_paths)}] Evaluating: {model_name}")
        print(f"Eval dir: {eval_dir}")
        try:
            result = evaluate_one(model_path, eval_dir, device)
            results.append(result)
            print(
                "loss={:.6f} acc={:.6f} prec={:.6f} rec={:.6f} f1={:.6f}".format(
                    result["loss"], result["accuracy"], result["precision"], result["recall"], result["f1"]
                )
            )
        except Exception as e:
            results.append({
                "model_name": model_name,
                "model_path": model_path,
                "arch": None,
                "eval_dir": eval_dir,
                "status": "error",
                "error": str(e),
                "per_class_f1": {},
            })
            print("ERROR:", str(e))

    output_path = resolve_output_path()
    write_results(results, output_path)
    print("\nSaved results to:", output_path)


def main():
    device = torch.device("cuda:" + args.GPU if torch.cuda.is_available() else "cpu")
    if args.batch_redo:
        run_batch_redo(device)
    else:
        run_single(device)


if __name__ == "__main__":
    main()
