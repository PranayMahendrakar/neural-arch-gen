# 🧠 Neural Architecture Generator

> Automatically designs PyTorch neural network architectures from problem type, dataset size, and hardware constraints.

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-green?logo=github)](https://pranaymahendrakar.github.io/neural-arch-gen/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🚀 Live Demo

**GitHub Pages:** [https://pranaymahendrakar.github.io/neural-arch-gen/](https://pranaymahendrakar.github.io/neural-arch-gen/)

The pages display:
- Generated neural network architectures (layer-by-layer)
- Training results and loss/accuracy charts
- Hyperparameter configurations
- Architecture search methodology documentation

---

## 📦 Project Structure

```
neural-arch-gen/
├── problem_analyzer.py        # Module 1: Analyses problem spec → AnalysisResult
├── architecture_generator.py  # Module 2: Builds PyTorch model → ArchSummary
├── training_script_builder.py # Module 3: Generates TrainingConfig + train.py
├── evaluation_system.py       # Module 4: Metrics, history tracking, reports
├── neural_arch_gen.py         # Orchestrator: wires all modules + CLI
├── docs/
│   └── index.html             # GitHub Pages site
└── README.md
```

---

## 🔧 Inputs → Outputs

### Inputs
| Parameter | Type | Description |
|-----------|------|-------------|
| `problem_type` | str | classification, regression, segmentation, detection, nlp, timeseries |
| `dataset_size` | int | Number of training samples |
| `hardware` | str | cpu / single_gpu / multi_gpu |
| `num_classes` | int | Number of output classes (optional) |
| `input_shape` | tuple | Input tensor shape, e.g. (3, 32, 32) |
| `gpu_memory_gb` | float | Available GPU VRAM in GB |

### Outputs
| Output | Description |
|--------|-------------|
| **PyTorch Model** | `nn.Module` ready to train |
| **Architecture Summary** | Layer breakdown + parameter count |
| **train.py** | Complete, runnable training script |
| **TrainingConfig** | All hyperparameters as a dataclass |
| **EvaluationReport** | JSON + HTML training report |

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/PranayMahendrakar/neural-arch-gen.git
cd neural-arch-gen

# 2. Install PyTorch
pip install torch torchvision

# 3. Run the generator
python neural_arch_gen.py \
  --problem_type classification \
  --dataset_size 8000 \
  --num_classes 10 \
  --input_shape 3 32 32 \
  --hardware single_gpu \
  --gpu_memory_gb 8.0 \
  --output_dir ./output
```

### Programmatic Usage

```python
from problem_analyzer import ProblemSpec, ProblemAnalyzer
from architecture_generator import ArchitectureGenerator
from training_script_builder import TrainingScriptBuilder
from evaluation_system import EvaluationSystem

# 1. Define your problem
spec = ProblemSpec(
    problem_type="classification",
    dataset_size=8000,
    num_classes=10,
    input_shape=(3, 32, 32),
    hardware="single_gpu",
    gpu_memory_gb=8.0,
)

# 2. Analyse
analysis = ProblemAnalyzer().analyze(spec)
print(f"Recommended arch: {analysis.arch_family} ({analysis.complexity_tier} tier)")

# 3. Generate model
model, summary = ArchitectureGenerator().generate(spec, analysis)
print(f"Model: {summary.name}  Params: {summary.num_params:,}")

# 4. Build training script
builder = TrainingScriptBuilder()
cfg     = builder.build_config(spec, analysis)
script  = builder.build_script(spec, analysis, summary)
with open("train.py", "w") as f: f.write(script)

# 5. Track evaluation
evaluator = EvaluationSystem(summary.name, spec.problem_type, summary)
# ... train and call evaluator.log_epoch(epoch, tr_loss, val_loss, ...) each epoch
report = evaluator.build_report(spec, analysis)
print(EvaluationSystem.to_json(report))
```

---

## 🏗️ Architecture Families

| Family | Problem Type | Trigger Condition |
|--------|-------------|-------------------|
| **AutoCNN** | Image Classification | dataset_size < 50k |
| **AutoResNet** | Image Classification | dataset_size ≥ 50k |
| **AutoUNet** | Segmentation | problem_type == segmentation |
| **AutoTransformer** | NLP | problem_type == nlp |
| **AutoLSTM / AutoTCN** | Time Series | problem_type == timeseries |
| **AutoMLP** | Regression / Tabular | problem_type == regression |

---

## 🧪 Supported Problem Types

- **classification** — Image, tabular, multi-class
- **regression** — Continuous output, tabular data
- **segmentation** — Pixel-wise labelling
- **detection** — Object detection (FPN backbone)
- **nlp** — Text classification, sentiment
- **timeseries** — Sequence forecasting / classification

---

## 📊 Architecture Search Methodology

The system uses **Rule-Based Neural Architecture Search (RBNAS)**:

1. **Problem Classification** — Map (problem_type, dataset_size) to complexity tier (tiny/small/medium/large) using fixed thresholds at 1k, 10k, 100k samples.
2. **Family Selection** — Select architecture family based on problem type + tier.
3. **Depth & Width Scaling** — Scale layer counts and filter sizes proportionally to the tier.
4. **Config Derivation** — Select optimizer, LR, scheduler, regularisation, and batch size from expert lookup tables.

This approach runs in **<1 second** with no GPU needed, making it ideal for rapid prototyping before committing to computationally expensive NAS methods.

---

## 📋 Module API Reference

### `ProblemAnalyzer`
```python
analyzer = ProblemAnalyzer()
result: AnalysisResult = analyzer.analyze(spec: ProblemSpec)
# result.arch_family, .complexity_tier, .recommended_batch_size,
# .use_pretrained, .use_mixed_precision, .regularization, .notes
```

### `ArchitectureGenerator`
```python
gen = ArchitectureGenerator()
model, summary = gen.generate(spec, analysis)
# model: nn.Module | summary: ArchSummary (name, family, num_params, layers, ...)
```

### `TrainingScriptBuilder`
```python
builder = TrainingScriptBuilder()
cfg: TrainingConfig = builder.build_config(spec, analysis)
script: str         = builder.build_script(spec, analysis, summary)
```

### `EvaluationSystem`
```python
ev = EvaluationSystem(model_name, problem_type, arch_summary)
ev.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, lr)
report: EvaluationReport = ev.build_report(spec, analysis, test_metrics)
json_str  = EvaluationSystem.to_json(report, path="report.json")
html_frag = EvaluationSystem.to_html_fragment(report)
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Generated by [neural-arch-gen](https://github.com/PranayMahendrakar/neural-arch-gen)*
