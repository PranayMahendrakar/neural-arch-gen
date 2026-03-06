#!/usr/bin/env python3
"""
neural_arch_gen.py  –  Main Orchestrator
=========================================
Single entry point that wires all four modules together:
  problem_analyzer  ->  architecture_generator
  ->  training_script_builder  ->  evaluation_system

Usage
-----
python neural_arch_gen.py \
    --problem_type classification \
    --dataset_size 8000 \
    --num_classes 10 \
    --input_shape 3 32 32 \
    --hardware single_gpu \
    --gpu_memory_gb 8.0 \
    --output_dir ./output
"""

import argparse
import json
import os
import sys

from problem_analyzer      import ProblemSpec, ProblemAnalyzer
from architecture_generator import ArchitectureGenerator
from training_script_builder import TrainingScriptBuilder
from evaluation_system       import EvaluationSystem


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class NeuralArchGen:
    def __init__(self):
        self.analyzer  = ProblemAnalyzer()
        self.gen       = ArchitectureGenerator()
        self.builder   = TrainingScriptBuilder()

    def run(self, spec: ProblemSpec, output_dir: str = "./output") -> dict:
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. Analyse problem ─────────────────────────────────────────
        print("[1/4] Analysing problem...")
        analysis = self.analyzer.analyze(spec)
        print(f"      arch_family={analysis.arch_family}  tier={analysis.complexity_tier}"
              f"  batch={analysis.recommended_batch_size}  pretrained={analysis.use_pretrained}")
        for note in analysis.notes:
            print(f"      NOTE: {note}")

        # ── 2. Generate architecture ───────────────────────────────────
        print("[2/4] Generating architecture...")
        model, arch_summary = self.gen.generate(spec, analysis)
        print(f"      Model: {arch_summary.name}  Params: {arch_summary.num_params:,}")
        for layer in arch_summary.layers:
            print(f"        - {layer}")

        # ── 3. Build training config & script ─────────────────────────
        print("[3/4] Building training pipeline...")
        cfg    = self.builder.build_config(spec, analysis)
        script = self.builder.build_script(spec, analysis, arch_summary)
        script_path = os.path.join(output_dir, "train.py")
        with open(script_path, "w") as f: f.write(script)
        print(f"      Training script written -> {script_path}")
        print(f"      Optimizer={cfg.optimizer}  LR={cfg.learning_rate}  Epochs={cfg.epochs}")

        # ── 4. Build evaluation framework ─────────────────────────────
        print("[4/4] Setting up evaluation system...")
        evaluator = EvaluationSystem(
            model_name   = arch_summary.name,
            problem_type = spec.problem_type,
            arch_summary = arch_summary,
        )

        # ── Compile results ────────────────────────────────────────────
        result = {
            "spec": {
                "problem_type":  spec.problem_type,
                "dataset_size":  spec.dataset_size,
                "num_classes":   spec.num_classes,
                "input_shape":   list(spec.input_shape),
                "hardware":      spec.hardware,
                "gpu_memory_gb": spec.gpu_memory_gb,
            },
            "analysis": {
                "arch_family":           analysis.arch_family,
                "complexity_tier":       analysis.complexity_tier,
                "recommended_batch_size": analysis.recommended_batch_size,
                "use_pretrained":        analysis.use_pretrained,
                "use_mixed_precision":   analysis.use_mixed_precision,
                "regularization":        analysis.regularization,
                "notes":                 analysis.notes,
            },
            "architecture": {
                "name":        arch_summary.name,
                "family":      arch_summary.family,
                "num_params":  arch_summary.num_params,
                "layers":      arch_summary.layers,
                "input_shape": list(arch_summary.input_shape),
                "output_shape": [str(x) for x in arch_summary.output_shape],
                "notes":       arch_summary.notes,
            },
            "training_config": cfg.__dict__,
        }

        # ── Save JSON summary ──────────────────────────────────────────
        summary_path = os.path.join(output_dir, "arch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSummary written -> {summary_path}")
        print(f"Training script -> {script_path}")

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Neural Architecture Generator")
    p.add_argument("--problem_type",  default="classification",
                   choices=["classification","regression","segmentation","detection","nlp","timeseries"])
    p.add_argument("--dataset_size",  type=int, default=10000)
    p.add_argument("--num_classes",   type=int, default=10)
    p.add_argument("--input_shape",   type=int, nargs="+", default=[3,32,32])
    p.add_argument("--hardware",      default="cpu",
                   choices=["cpu","single_gpu","multi_gpu"])
    p.add_argument("--gpu_memory_gb", type=float, default=8.0)
    p.add_argument("--latency_critical", action="store_true")
    p.add_argument("--output_dir",    default="./output")
    return p.parse_args()


def main():
    args = parse_args()
    spec = ProblemSpec(
        problem_type      = args.problem_type,
        dataset_size      = args.dataset_size,
        num_classes       = args.num_classes,
        input_shape       = tuple(args.input_shape),
        hardware          = args.hardware,
        gpu_memory_gb     = args.gpu_memory_gb,
        latency_critical  = args.latency_critical,
    )
    nag    = NeuralArchGen()
    result = nag.run(spec, output_dir=args.output_dir)
    print("\n=== Architecture Summary ===")
    print(json.dumps(result["architecture"], indent=2))
    return result


if __name__ == "__main__":
    main()
