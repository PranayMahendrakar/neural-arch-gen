"""
evaluation_system.py
Provides evaluation metrics, a training-history tracker,
and utilities to generate JSON/HTML reports for GitHub Pages.
"""

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-epoch record
# ---------------------------------------------------------------------------

@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float = 0.0
    val_acc: float = 0.0
    lr: float = 0.0
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

class Metrics:
    @staticmethod
    def accuracy(preds, targets) -> float:
        correct = sum(p == t for p, t in zip(preds, targets))
        return correct / len(targets) if targets else 0.0

    @staticmethod
    def top_k_accuracy(logits_list, targets, k=5) -> float:
        correct = 0
        for logits, t in zip(logits_list, targets):
            top_k = sorted(range(len(logits)), key=lambda i: -logits[i])[:k]
            correct += int(t in top_k)
        return correct / len(targets) if targets else 0.0

    @staticmethod
    def mse(preds, targets) -> float:
        return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)

    @staticmethod
    def mae(preds, targets) -> float:
        return sum(abs(p - t) for p, t in zip(preds, targets)) / len(targets)

    @staticmethod
    def rmse(preds, targets) -> float:
        return math.sqrt(Metrics.mse(preds, targets))

    @staticmethod
    def f1_binary(preds, targets) -> float:
        tp = sum(p == 1 and t == 1 for p, t in zip(preds, targets))
        fp = sum(p == 1 and t == 0 for p, t in zip(preds, targets))
        fn = sum(p == 0 and t == 1 for p, t in zip(preds, targets))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    @staticmethod
    def iou_binary(pred_mask, true_mask) -> float:
        """Flat lists of 0/1 predictions."""
        intersection = sum(p & t for p, t in zip(pred_mask, true_mask))
        union        = sum(p | t for p, t in zip(pred_mask, true_mask))
        return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Evaluation system
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    model_name: str
    problem_type: str
    arch_family: str
    num_params: int
    complexity_tier: str
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    total_epochs_trained: int
    training_time_sec: float
    history: List[EpochRecord] = field(default_factory=list)
    final_test_metrics: Dict[str, float] = field(default_factory=dict)
    hardware: str = "cpu"
    notes: str = ""


class EvaluationSystem:
    """Tracks training progress and produces structured reports."""

    def __init__(self, model_name: str, problem_type: str, arch_summary: Any):
        self.model_name    = model_name
        self.problem_type  = problem_type
        self.arch_summary  = arch_summary
        self.history: List[EpochRecord] = []
        self._start_time   = time.time()
        self._best_val_loss = float("inf")
        self._best_epoch   = 0

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_acc: float = 0.0, val_acc: float = 0.0, lr: float = 0.0):
        elapsed = time.time() - self._start_time
        rec = EpochRecord(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                          train_acc=train_acc, val_acc=val_acc, lr=lr, elapsed_sec=elapsed)
        self.history.append(rec)
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_epoch    = epoch

    def build_report(self, spec: Any, analysis: Any,
                     test_metrics: Optional[Dict[str, float]] = None) -> EvaluationReport:
        best_rec = max(self.history, key=lambda r: r.val_acc) if self.history else None
        return EvaluationReport(
            model_name           = self.model_name,
            problem_type         = self.problem_type,
            arch_family          = self.arch_summary.family,
            num_params           = self.arch_summary.num_params,
            complexity_tier      = analysis.complexity_tier,
            best_val_loss        = self._best_val_loss,
            best_val_acc         = best_rec.val_acc if best_rec else 0.0,
            best_epoch           = self._best_epoch,
            total_epochs_trained = len(self.history),
            training_time_sec    = time.time() - self._start_time,
            history              = self.history,
            final_test_metrics   = test_metrics or {},
            hardware             = spec.hardware,
        )

    # ------------------------------------------------------------------
    # Report serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(report: EvaluationReport, path: Optional[str] = None) -> str:
        def default_encoder(obj):
            if isinstance(obj, EpochRecord): return asdict(obj)
            return str(obj)
        data = {
            "model_name":           report.model_name,
            "problem_type":         report.problem_type,
            "arch_family":          report.arch_family,
            "num_params":           report.num_params,
            "complexity_tier":      report.complexity_tier,
            "best_val_loss":        report.best_val_loss,
            "best_val_acc":         report.best_val_acc,
            "best_epoch":           report.best_epoch,
            "total_epochs_trained": report.total_epochs_trained,
            "training_time_sec":    report.training_time_sec,
            "final_test_metrics":   report.final_test_metrics,
            "hardware":             report.hardware,
            "history":              [asdict(r) for r in report.history],
        }
        result = json.dumps(data, indent=2, default=default_encoder)
        if path:
            with open(path, "w") as f: f.write(result)
        return result

    @staticmethod
    def to_html_fragment(report: EvaluationReport) -> str:
        rows = "".join(
            f"<tr><td>{r.epoch}</td><td>{r.train_loss:.4f}</td><td>{r.val_loss:.4f}</td>"
            f"<td>{r.train_acc:.3f}</td><td>{r.val_acc:.3f}</td><td>{r.lr:.2e}</td></tr>"
            for r in report.history
        )
        return f"""
<div class="report-card" id="report-{report.model_name.lower().replace(' ','-')}">
  <h3>{report.model_name}</h3>
  <ul>
    <li><b>Problem:</b> {report.problem_type}</li>
    <li><b>Architecture family:</b> {report.arch_family}</li>
    <li><b>Parameters:</b> {report.num_params:,}</li>
    <li><b>Complexity tier:</b> {report.complexity_tier}</li>
    <li><b>Best val loss:</b> {report.best_val_loss:.4f} (epoch {report.best_epoch})</li>
    <li><b>Best val accuracy:</b> {report.best_val_acc:.3f}</li>
    <li><b>Total epochs:</b> {report.total_epochs_trained}</li>
    <li><b>Training time:</b> {report.training_time_sec:.1f}s</li>
    <li><b>Hardware:</b> {report.hardware}</li>
  </ul>
  <table>
    <thead><tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Train Acc</th><th>Val Acc</th><th>LR</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from problem_analyzer import ProblemSpec, ProblemAnalyzer
    from architecture_generator import ArchitectureGenerator

    spec     = ProblemSpec(problem_type="classification", dataset_size=8000,
                           num_classes=10, input_shape=(3,32,32), hardware="single_gpu")
    analysis = ProblemAnalyzer().analyze(spec)
    _, summary = ArchitectureGenerator().generate(spec, analysis)

    evaluator = EvaluationSystem(model_name=summary.name, problem_type=spec.problem_type, arch_summary=summary)
    # Simulate a few epochs
    import random
    for ep in range(1, 6):
        evaluator.log_epoch(ep, train_loss=1.0/ep + random.random()*0.1,
                            val_loss=1.1/ep + random.random()*0.1,
                            train_acc=0.5+ep*0.08, val_acc=0.48+ep*0.07, lr=1e-3)

    report = evaluator.build_report(spec, analysis, test_metrics={"test_acc": 0.82, "test_loss": 0.51})
    print(EvaluationSystem.to_json(report))
