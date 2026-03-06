"""
architecture_generator.py
Generates PyTorch neural network architectures based on AnalysisResult
from problem_analyzer.  Each builder returns (model, arch_summary_dict).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class ArchSummary:
    name: str
    family: str
    num_params: int
    layers: list
    input_shape: tuple
    output_shape: tuple
    notes: str = ""


# ---------------------------------------------------------------------------
# Base builder
# ---------------------------------------------------------------------------

class _BaseBuilder:
    def build(self, spec: Any, analysis: Any) -> Tuple[nn.Module, ArchSummary]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MLP  (regression / tiny classification)
# ---------------------------------------------------------------------------

class MLPBuilder(_BaseBuilder):
    def build(self, spec, analysis):
        in_features = 1
        for d in spec.input_shape:
            in_features *= d

        hidden = self._hidden_sizes(spec.dataset_size, analysis.complexity_tier)
        layers = []
        prev = in_features
        layer_descs = []
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            if "dropout" in analysis.regularization:
                layers.append(nn.Dropout(0.3))
            layer_descs.append(f"Linear({prev}->{h}) + BN + ReLU")
            prev = h

        out_features = spec.num_classes if spec.num_classes else 1
        layers.append(nn.Linear(prev, out_features))
        layer_descs.append(f"Linear({prev}->{out_features})")

        model = nn.Sequential(*layers)
        params = sum(p.numel() for p in model.parameters())
        return model, ArchSummary(
            name="AutoMLP", family="mlp", num_params=params,
            layers=layer_descs, input_shape=spec.input_shape,
            output_shape=(out_features,),
            notes=f"Depth={len(hidden)+1}, hidden={hidden}"
        )

    @staticmethod
    def _hidden_sizes(n, tier):
        if tier == "tiny":  return [64, 32]
        if tier == "small": return [256, 128, 64]
        if tier == "medium": return [512, 256, 128]
        return [1024, 512, 256, 128]


# ---------------------------------------------------------------------------
# CNN  (small image classification)
# ---------------------------------------------------------------------------

class CNNBuilder(_BaseBuilder):
    def build(self, spec, analysis):
        channels, h, w = spec.input_shape
        num_classes = spec.num_classes or 10
        tier = analysis.complexity_tier
        cfg = {"tiny": [32,64], "small": [32,64,128], "medium": [64,128,256], "large": [64,128,256,512]}.get(tier,[64,128,256])

        layers = []
        layer_descs = []
        prev_c = channels
        for out_c in cfg:
            layers += [nn.Conv2d(prev_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(), nn.MaxPool2d(2)]
            layer_descs.append(f"Conv2d({prev_c}->{out_c},3x3) + BN + ReLU + MaxPool2")
            prev_c = out_c
            h, w = h//2, w//2

        layers += [nn.AdaptiveAvgPool2d((4,4)), nn.Flatten()]
        flat = prev_c * 4 * 4
        fc_hidden = 256 if tier != "tiny" else 128
        layers += [nn.Linear(flat, fc_hidden), nn.ReLU()]
        if "dropout" in analysis.regularization:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(fc_hidden, num_classes))

        model = nn.Sequential(*layers)
        params = sum(p.numel() for p in model.parameters())
        return model, ArchSummary(
            name="AutoCNN", family="cnn", num_params=params,
            layers=layer_descs + [f"Flatten->FC({flat}->{fc_hidden})->FC({fc_hidden}->{num_classes})"],
            input_shape=spec.input_shape, output_shape=(num_classes,),
            notes=f"Conv filters: {cfg}"
        )


# ---------------------------------------------------------------------------
# ResNet-style  (medium/large classification)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c)
        )
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.net(x) + x)

class ResNetBuilder(_BaseBuilder):
    def build(self, spec, analysis):
        channels = spec.input_shape[0]
        num_classes = spec.num_classes or 10
        stage_cfg = {"small":[1,1],"medium":[2,2,2],"large":[3,4,6,3]}.get(analysis.complexity_tier,[2,2,2])
        filter_cfg = [64,128,256,512][:len(stage_cfg)]

        layers, descs, prev_c = [nn.Conv2d(channels,64,7,stride=2,padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3,stride=2,padding=1)], ["Stem: Conv7x7-BN-ReLU-MaxPool"], 64
        for i,(n_blocks, out_c) in enumerate(zip(stage_cfg, filter_cfg)):
            if prev_c != out_c:
                layers.append(nn.Conv2d(prev_c, out_c, 1))
                prev_c = out_c
            for _ in range(n_blocks):
                layers.append(ResBlock(out_c))
            descs.append(f"Stage{i+1}: {n_blocks}x ResBlock({out_c})")

        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(prev_c, num_classes)]
        descs.append(f"GAP -> FC({prev_c}->{num_classes})")
        model = nn.Sequential(*layers)
        params = sum(p.numel() for p in model.parameters())
        return model, ArchSummary(
            name="AutoResNet", family="resnet", num_params=params,
            layers=descs, input_shape=spec.input_shape, output_shape=(num_classes,),
            notes=f"Stages: {list(zip(stage_cfg, filter_cfg))}"
        )


# ---------------------------------------------------------------------------
# U-Net (segmentation)
# ---------------------------------------------------------------------------

class UNetBuilder(_BaseBuilder):
    def build(self, spec, analysis):
        in_c = spec.input_shape[0]
        out_c = spec.num_classes or 2
        features = [64, 128, 256, 512] if analysis.complexity_tier in ("medium","large") else [32,64,128]

        class DoubleConv(nn.Module):
            def __init__(self, i, o):
                super().__init__()
                self.block = nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),
                                           nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU())
            def forward(self, x): return self.block(x)

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.downs = nn.ModuleList()
                self.ups   = nn.ModuleList()
                prev = in_c
                for f in features:
                    self.downs.append(DoubleConv(prev,f)); prev = f
                self.pool = nn.MaxPool2d(2)
                self.bottleneck = DoubleConv(features[-1], features[-1]*2)
                prev = features[-1]*2
                for f in reversed(features):
                    self.ups.append(nn.ConvTranspose2d(prev,f,2,stride=2))
                    self.ups.append(DoubleConv(f*2,f))
                    prev = f
                self.head = nn.Conv2d(features[0], out_c, 1)
            def forward(self, x):
                skips = []
                for d in self.downs:
                    x = d(x); skips.append(x); x = self.pool(x)
                x = self.bottleneck(x)
                skips = skips[::-1]
                for i in range(0, len(self.ups), 2):
                    x = self.ups[i](x)
                    s = skips[i//2]
                    if x.shape != s.shape:
                        x = torch.nn.functional.interpolate(x, size=s.shape[2:])
                    x = self.ups[i+1](torch.cat([s,x],1))
                return self.head(x)

        model = UNet()
        params = sum(p.numel() for p in model.parameters())
        descs = [f"Encoder: {features}", "Bottleneck", f"Decoder: {list(reversed(features))}", f"Head: Conv1x1->{out_c}"]
        return model, ArchSummary(
            name="AutoUNet", family="unet", num_params=params,
            layers=descs, input_shape=spec.input_shape, output_shape=(out_c,"H","W"),
            notes=f"Features: {features}"
        )


# ---------------------------------------------------------------------------
# LSTM (time series)
# ---------------------------------------------------------------------------

class LSTMBuilder(_BaseBuilder):
    def build(self, spec, analysis):
        input_size = spec.input_shape[-1]
        out = spec.num_classes or 1
        hidden = 128 if analysis.complexity_tier in ("medium","large") else 64
        num_layers = 2 if analysis.complexity_tier in ("medium","large") else 1

        class LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, dropout=0.2 if num_layers>1 else 0)
                self.fc = nn.Linear(hidden, out)
            def forward(self, x):
                out_seq, _ = self.lstm(x)
                return self.fc(out_seq[:,-1,:])

        model = LSTMNet()
        params = sum(p.numel() for p in model.parameters())
        return model, ArchSummary(
            name="AutoLSTM", family="lstm", num_params=params,
            layers=[f"LSTM(input={input_size},hidden={hidden},layers={num_layers})", f"FC({hidden}->{out})"],
            input_shape=spec.input_shape, output_shape=(out,),
            notes=f"Bidirectional=False, hidden={hidden}"
        )


# ---------------------------------------------------------------------------
# Transformer (NLP)
# ---------------------------------------------------------------------------

class TransformerBuilder(_BaseBuilder):
    def build(self, spec, analysis):
        vocab_size = getattr(spec, 'vocab_size', 30522)
        max_len    = getattr(spec, 'max_len', 128)
        num_classes = spec.num_classes or 2
        d_model = 128 if analysis.complexity_tier in ("tiny","small") else 256
        nhead   = 4 if d_model == 128 else 8
        n_layers= 2 if analysis.complexity_tier in ("tiny","small") else 4

        class TransformerClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.pos   = nn.Embedding(max_len, d_model)
                enc_layer  = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
                self.enc   = nn.TransformerEncoder(enc_layer, n_layers)
                self.fc    = nn.Linear(d_model, num_classes)
            def forward(self, x):
                pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                h = self.embed(x) + self.pos(pos)
                h = self.enc(h)
                return self.fc(h.mean(1))

        model = TransformerClassifier()
        params = sum(p.numel() for p in model.parameters())
        return model, ArchSummary(
            name="AutoTransformer", family="transformer", num_params=params,
            layers=[f"Embedding({vocab_size},{d_model})", f"PositionalEmbedding(max={max_len})",
                    f"{n_layers}x TransformerEncoderLayer(d={d_model},heads={nhead})", f"MeanPool->FC->{num_classes}"],
            input_shape=spec.input_shape, output_shape=(num_classes,),
            notes=f"d_model={d_model}, heads={nhead}, layers={n_layers}"
        )


# ---------------------------------------------------------------------------
# Registry & main entrypoint
# ---------------------------------------------------------------------------

BUILDERS = {
    "mlp":         MLPBuilder,
    "cnn":         CNNBuilder,
    "resnet":      ResNetBuilder,
    "unet":        UNetBuilder,
    "lstm":        LSTMBuilder,
    "tcn":         LSTMBuilder,   # TCN reuses LSTM builder as placeholder
    "transformer": TransformerBuilder,
    "fpn":         CNNBuilder,    # FPN falls back to CNN builder
}


class ArchitectureGenerator:
    def generate(self, spec, analysis) -> Tuple[nn.Module, ArchSummary]:
        family = analysis.arch_family
        builder_cls = BUILDERS.get(family, MLPBuilder)
        return builder_cls().build(spec, analysis)


if __name__ == "__main__":
    from problem_analyzer import ProblemSpec, ProblemAnalyzer
    spec = ProblemSpec(problem_type="classification", dataset_size=8000,
                       num_classes=10, input_shape=(3,32,32), hardware="single_gpu")
    analysis = ProblemAnalyzer().analyze(spec)
    gen = ArchitectureGenerator()
    model, summary = gen.generate(spec, analysis)
    print(f"Model: {summary.name}  Params: {summary.num_params:,}")
    print("Layers:", summary.layers)
