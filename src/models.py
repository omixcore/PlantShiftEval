"""Model zoo for the PlantShiftEval methodological study: baselines + MTD-Net under test.

MTD-Net (Multi-scale Texture-Decoupled Network)
------------------------------------------------
Motivation.  In leaf-disease images two factors are entangled: *crop identity*
(global shape, colour, background) and *disease* (local, multi-scale lesion
texture).  Crop/background shortcuts are one hypothesis for transfer errors; the present
architecture and diagnostics do not by themselves establish that mechanism.

Design.
  1. Appearance path  -- aggressively downsampled, small receptive-field stack,
     produces a global descriptor that is explicitly supervised to predict the
     crop species (auxiliary multi-task head).
  2. Texture path     -- stays at H/4 with depthwise-separable small kernels and
     dilated blocks, preserving lesion edges and spot granularity.
  3. Cross-scale gate -- the appearance descriptor emits both a channel gate and
     a coarse spatial prior that modulate the texture features, i.e. species-
     conditioned lesion attention.
  4. Disease head sees the gated texture descriptor concatenated with the
     appearance descriptor, so both descriptors enter the classifier; semantic disentanglement is not guaranteed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class DSConv(nn.Module):
    """Depthwise-separable convolution."""

    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, d, dilation=d, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class DSBlock(nn.Module):
    def __init__(self, cin: int, cout: int, s: int = 1, d: int = 1):
        super().__init__()
        self.c1 = DSConv(cin, cout, 3, s, d)
        self.c2 = DSConv(cout, cout, 3, 1, d)
        self.short = None
        if s != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, s, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.c2(self.c1(x))
        r = x if self.short is None else self.short(x)
        return F.silu(y + r)


class CrossScaleGate(nn.Module):
    """Turn a global appearance vector into a channel gate + spatial prior."""

    def __init__(self, c_app: int, c_det: int):
        super().__init__()
        self.ch = nn.Sequential(nn.Linear(c_app, c_det), nn.Sigmoid())
        self.sp = nn.Linear(c_app, 36)
        self.proj = nn.Conv2d(1, 1, 1)

    def forward(self, g, d):
        ch = self.ch(g)[:, :, None, None]
        b = g.shape[0]
        s = self.sp(g).view(b, 1, 6, 6)
        s = F.interpolate(s, size=d.shape[-2:], mode="bilinear", align_corners=False)
        return d * ch * torch.sigmoid(self.proj(s)), torch.sigmoid(s)


class MTDNet(nn.Module):
    def __init__(self, n_disease: int, n_species: int, w: int = 32,
                 c_det: int = 96, drop: float = 0.2, g_init: float = 0.02):
        super().__init__()
        self.g_init = g_init
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.SiLU(inplace=True),
            DSConv(w, w, 3, 2),
        )
        # appearance path: H/8 -> H/32
        self.app = nn.Sequential(
            DSBlock(w, 2 * w, s=2),
            DSBlock(2 * w, 4 * w, s=2),
            DSBlock(4 * w, 8 * w, s=2),
        )
        self.c_app = 8 * w
        # texture path: kept at H/4
        self.det = nn.Sequential(
            DSBlock(w, c_det, s=1, d=1),
            DSBlock(c_det, c_det, s=1, d=1),
            DSBlock(c_det, c_det, s=1, d=2),
            DSBlock(c_det, c_det, s=1, d=2),
        )
        self.gate = CrossScaleGate(self.c_app, c_det)
        self.species_head = nn.Linear(self.c_app, n_species)
        self.disease_head = nn.Sequential(
            nn.Linear(self.c_app + c_det, 256), nn.SiLU(inplace=True),
            nn.Dropout(drop), nn.Linear(256, n_disease),
        )
        self._init_weights()
        # The gate ends in sigmoid(Linear(.)) with trunc_normal(std=0.02)
        # weights, i.e. it starts at a flat 0.25 multiplier with almost no
        # channel selectivity.  Whether a short training budget changes this
        # initial scaling requires direct measurement.
        # A larger init is a sensitivity control, not a proven improvement.
        if g_init != 0.02:
            for lin in (self.gate.ch[0], self.gate.sp):
                nn.init.trunc_normal_(lin.weight, std=g_init)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.stem(x)
        app = self.app(f)
        g = F.adaptive_avg_pool2d(app, 1).flatten(1)
        det = self.det(f)
        det, gate = self.gate(g, det)
        d = F.adaptive_avg_pool2d(det, 1).flatten(1)
        return {
            "disease": self.disease_head(torch.cat([d, g], 1)),
            "species": self.species_head(g),
            "gate": gate,
            "detail_feat": det,
            "app_feat": g,
        }


class MTDNetPT(MTDNet):
    """MTD-Net on top of an ImageNet-pretrained ResNet18 trunk.

    Why this variant exists
    -----------------------
    Historical cross-architecture comparisons did not isolate pretraining.
    This variant instead asks a conditional question:
    is the decoupled design *still* useful once both models start from the same
    pre-trained features?  MTD-P answers it by cutting ResNet18 at H/4:

        conv1/bn/relu/maxpool/layer1 -> H/4, 64ch  ---+--> texture branch (H/4)
                                                      |
                                   layer2/layer3/layer4 -> appearance path

    Everything else (gate, species head, disease head) is identical to MTD-Net,
    against ResNet18 this adds a high-resolution branch, gate, auxiliary head
    and a different disease classifier, changing capacity and compute.  Pre-trained weights are restored
    after initialisation, so the comparison is "same init, different wiring".
    """

    def __init__(self, n_disease: int, n_species: int, c_det: int = 96,
                 drop: float = 0.2):
        nn.Module.__init__(self)
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # clone: state_dict() returns references, and _init_weights() rewrites
        # those same tensors in place, which would silently destroy the weights
        pretrained = {k: v.detach().clone() for k, v in r.state_dict().items()}
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.app = nn.Sequential(r.layer2, r.layer3, r.layer4)  # H/8 -> H/32
        self.c_app = 512
        self.det = nn.Sequential(
            DSBlock(64, c_det, s=1, d=1),
            DSBlock(c_det, c_det, s=1, d=1),
            DSBlock(c_det, c_det, s=1, d=2),
            DSBlock(c_det, c_det, s=1, d=2),
        )
        self.gate = CrossScaleGate(self.c_app, c_det)
        self.species_head = nn.Linear(self.c_app, n_species)
        self.disease_head = nn.Sequential(
            nn.Linear(self.c_app + c_det, 256), nn.SiLU(inplace=True),
            nn.Dropout(drop), nn.Linear(256, n_disease),
        )
        self._init_weights()
        self._restore_pretrained(pretrained)

    def _restore_pretrained(self, sd: dict):
        """Map torchvision keys onto stem/app and reload (init overwrote them)."""
        new = {}
        for k, v in sd.items():
            if k.startswith("conv1.") or k.startswith("bn1."):
                new["stem." + k.replace("conv1", "0").replace("bn1", "1")] = v
            elif k.startswith("layer1."):
                new["stem.4." + k[len("layer1."):]] = v
            elif k.startswith("layer2."):
                new["app.0." + k[len("layer2."):]] = v
            elif k.startswith("layer3."):
                new["app.1." + k[len("layer3."):]] = v
            elif k.startswith("layer4."):
                new["app.2." + k[len("layer4."):]] = v
        missing, unexpected = self.load_state_dict(new, strict=False)
        assert not unexpected, unexpected[:5]
        # only the freshly added texture/gate/head modules may be "missing"
        allowed = ("det.", "gate.", "species_head.", "disease_head.")
        left = [m for m in missing if not m.startswith(allowed)]
        assert not left, f"pretrained weights not restored: {left[:5]}"
        self.pretrained_keys = len(new)


class MTDConcat(MTDNet):
    """Ablation: remove the cross-scale gate (plain concatenation)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = None

    def forward(self, x):
        f = self.stem(x)
        app = self.app(f)
        g = F.adaptive_avg_pool2d(app, 1).flatten(1)
        det = self.det(f)
        d = F.adaptive_avg_pool2d(det, 1).flatten(1)
        return {"disease": self.disease_head(torch.cat([d, g], 1)),
                "species": self.species_head(g),
                "gate": None, "detail_feat": det, "app_feat": g}


class MTDSingle(MTDNet):
    """Ablation: single-task (no species auxiliary head)."""

    def forward(self, x):
        out = super().forward(x)
        out["species"] = None
        return out


class MTDAppearanceOnly(MTDNet):
    """Remove the high-resolution branch; capacity also changes and is reported."""

    def __init__(self, n_disease, n_species):
        super().__init__(n_disease, n_species)
        self.det = None
        self.gate = None
        self.disease_head = nn.Sequential(
            nn.Linear(self.c_app, 256), nn.SiLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, n_disease))

    def forward(self, x):
        app = self.app(self.stem(x))
        g = F.adaptive_avg_pool2d(app, 1).flatten(1)
        return {"disease": self.disease_head(g), "species": self.species_head(g),
                "gate": None, "detail_feat": None, "app_feat": g}


class MTDNetGate(MTDNet):
    """Same wiring, but the cross-scale gate is initialised with a usable
    dynamic range instead of a flat 0.25 multiplier.

    Motivation: the gate ablation removed a module that, at its default init,
    was numerically almost an identity scaling, so "no effect" was the only
    possible outcome.  Re-running the ablation with a gate that can actually
    select channels is the fair test of whether cross-scale gating helps.
    """

    def __init__(self, n_disease: int, n_species: int, g_init: float = 0.15, **kw):
        super().__init__(n_disease, n_species, g_init=g_init, **kw)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class _OutWrap(nn.Module):
    """Give third-party backbones the same dict interface as MTD-Net."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return {"disease": self.net(x), "species": None, "gate": None,
                "detail_feat": None, "app_feat": None}


def build_baseline(name: str, n_disease: int, n_species: int | None = None,
                   pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_disease)
        return _OutWrap(m)
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_disease)
        return _OutWrap(m)
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_disease)
        return _OutWrap(m)
    if name == "smallcnn":
        return SmallCNN(n_disease)
    raise ValueError(f"unknown baseline {name}")


class SmallCNN(nn.Module):
    """From-scratch reference with a comparable parameter budget."""

    def __init__(self, n_disease: int, w: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            DSConv(3, w, 3, 2), DSBlock(w, 2 * w, 2), DSBlock(2 * w, 4 * w, 2),
            DSBlock(4 * w, 8 * w, 2), DSBlock(8 * w, 8 * w, 2),
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(8 * w, n_disease))

    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)
        return {"disease": self.head(f), "species": None, "gate": None,
                "detail_feat": None, "app_feat": f}


MODEL_ZOO = {
    "smallcnn": "scratch",
    "resnet18": "pretrained",
    "mobilenet_v3_small": "pretrained",
    "efficientnet_b0": "pretrained",
    "mtdnet": "scratch",
    "mtdnet_pt": "pretrained",
    "mtd_concat": "scratch",
    "mtd_single": "scratch",
    "mtdnet_g": "scratch",
}


def build_model(name: str, n_disease: int, n_species: int) -> nn.Module:
    if name == "mtdnet":
        return MTDNet(n_disease, n_species)
    if name == "mtdnet_pt":
        return MTDNetPT(n_disease, n_species)
    if name == "mtd_concat":
        return MTDConcat(n_disease, n_species)
    if name == "mtd_single":
        return MTDSingle(n_disease, n_species)
    if name == "mtd_appearance":
        return MTDAppearanceOnly(n_disease, n_species)
    if name == "mtdnet_g":
        return MTDNetGate(n_disease, n_species)
    return build_baseline(name, n_disease, pretrained=(MODEL_ZOO[name] == "pretrained"))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
