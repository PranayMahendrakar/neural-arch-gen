"""
problem_analyzer.py
Analyzes the ML problem type, dataset size, and hardware constraints
to determine the best neural network family and training strategy.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProblemSpec:
    problem_type: str          # classification | regression | segmentation | detection | nlp | timeseries
    dataset_size: int          # number of samples
    num_classes: Optional[int] = None
    input_shape: tuple = (3, 224, 224)
    hardware: str = "cpu"      # cpu | single_gpu | multi_gpu
    gpu_memory_gb: float = 8.0
    latency_critical: bool = False


@dataclass
class AnalysisResult:
    arch_family: str
    complexity_tier: str       # tiny | small | medium | large
    recommended_batch_size: int
    use_pretrained: bool
    use_mixed_precision: bool
    regularization: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class ProblemAnalyzer:
    """Rule-based expert system that maps problem characteristics to architecture recommendations."""

    COMPLEXITY_THRESHOLDS = {
        "tiny":   1_000,
        "small":  10_000,
        "medium": 100_000,
        "large":  math.inf,
    }

    def analyze(self, spec: ProblemSpec) -> AnalysisResult:
        complexity = self._dataset_complexity(spec.dataset_size)
        arch_family = self._pick_arch_family(spec)
        batch_size  = self._recommend_batch_size(spec, arch_family)
        use_pretrained = self._should_use_pretrained(spec, complexity)
        use_amp = spec.hardware in ("single_gpu", "multi_gpu")
        regularization = self._regularization_strategy(spec, complexity)
        notes = self._build_notes(spec, complexity, arch_family)
        return AnalysisResult(
            arch_family=arch_family,
            complexity_tier=complexity,
            recommended_batch_size=batch_size,
            use_pretrained=use_pretrained,
            use_mixed_precision=use_amp,
            regularization=regularization,
            notes=notes,
        )

    def _dataset_complexity(self, n: int) -> str:
        for tier, threshold in self.COMPLEXITY_THRESHOLDS.items():
            if n < threshold:
                return tier
        return "large"

    def _pick_arch_family(self, spec: ProblemSpec) -> str:
        mapping = {
            "classification": "cnn" if spec.dataset_size < 50_000 else "resnet",
            "regression":     "mlp",
            "segmentation":   "unet",
            "detection":      "fpn",
            "nlp":            "transformer",
            "timeseries":     "lstm" if spec.dataset_size < 20_000 else "tcn",
        }
        return mapping.get(spec.problem_type.lower(), "mlp")

    def _recommend_batch_size(self, spec: ProblemSpec, family: str) -> int:
        base = {"mlp": 256,"cnn": 128,"resnet": 64,"unet": 8,"fpn": 4,"transformer": 32,"lstm": 64,"tcn": 64}.get(family, 64)
        if spec.hardware == "cpu": base = min(base, 32)
        elif spec.gpu_memory_gb <= 4: base = base // 2
        return max(base, 1)

    def _should_use_pretrained(self, spec: ProblemSpec, complexity: str) -> bool:
        if spec.problem_type.lower() in ("classification","segmentation","detection"):
            return complexity in ("tiny","small")
        return False

    def _regularization_strategy(self, spec: ProblemSpec, complexity: str) -> list:
        regs = []
        if complexity in ("tiny","small"): regs += ["dropout","weight_decay","early_stopping"]
        elif complexity == "medium": regs += ["dropout","weight_decay"]
        else: regs += ["weight_decay","label_smoothing"]
        if spec.problem_type.lower() in ("classification","segmentation"): regs.append("data_augmentation")
        return regs

    def _build_notes(self, spec: ProblemSpec, complexity: str, family: str) -> list:
        notes = []
        if complexity == "tiny": notes.append("Very small dataset - strong regularization & data augmentation recommended.")
        if spec.latency_critical: notes.append("Latency-critical: prefer lighter architectures and post-training quantization.")
        if spec.hardware == "cpu": notes.append("CPU training: keep model small and batch size low.")
        if family == "transformer" and spec.dataset_size < 5_000: notes.append("Transformer on tiny data: consider fine-tuning a pretrained LM instead.")
        return notes


if __name__ == "__main__":
    import json
    spec = ProblemSpec(problem_type="classification", dataset_size=8000, num_classes=10,
                       input_shape=(3,32,32), hardware="single_gpu", gpu_memory_gb=8.0)
    result = ProblemAnalyzer().analyze(spec)
    print(json.dumps(result.__dict__, indent=2, default=str))
