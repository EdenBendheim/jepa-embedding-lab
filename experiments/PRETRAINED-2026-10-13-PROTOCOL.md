# Frozen pretrained comparator — October 13, 2026

Declared before pretrained CIFAR-10 features or downstream scores are observed.

## Comparator and limits

Use torchvision 0.23 ResNet-18, explicitly named `IMAGENET1K_V1` weights,
with the ImageNet classification head removed. Frozen global-average features
have 512 dimensions. Use its documented deterministic tensor transform:
resize shorter edge to 256 with bilinear antialiasing, center crop 224, scale
to [0,1], and normalize RGB by the ImageNet mean/std.
[Official architecture, weights and transforms](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet18.html).

This is a transfer-learning reference, not a matched-training JEPA control.
It uses labeled ImageNet pretraining, more parameters, a different architecture,
224-pixel inputs and 512 features instead of the compact JEPA's 32. Report these
differences; do not attribute a gap solely to the learning objective or claim
external pretraining was free of CIFAR overlap without evidence.

## Data and probe protocol

Reuse the fixed October 9 pilot: 500 fitted training and 200 tuning-validation
examples. Reuse the 223-step default-mask mean JEPA, matching random and pixel
artifacts. Both partitions are already observed; this is exploratory
validation selection. No official test images or fresh October 12 confirmation
examples are used. Same ordered indices/labels, all eight predeclared AdamW
settings (LR 0.0001/0.001/0.01/0.05 x decay 0.01/0.1, 200 steps), seeds
29/31/37, and ridge alpha 0.01/0.1/1/10 for every representation. Normalize
features using fitted training rows only. Persist and reproduce every selected
classifier. Count extraction time separately from classifier fitting.

## Implementation and execution boundaries

- Optional pinned torchvision extra; synthetic checks do not download weights.
- Read explicitly supplied local weights with weights_only=True on CPU; verify
  exact bytes against an expected SHA-256 and reject incompatible/non-finite state.
- Deterministic transforms and frozen inference preserve caller RNG/mode.
- Artifact identity binds weights, transform, architecture, runtime and selected
  examples; current JEPA/pixel artifacts remain backward compatible.
- No downloading inside extraction, no refitting on confirmation, no paid compute.
- Use one local CPU thread, batch at most 16, and at most 1,000 selected images
  per partition. Raw images, weights, feature arrays and probes stay ignored.
- Publish aggregate accuracy, provenance, dimensions and limitations only.

A failed or incomplete run is reported as such. Final test evaluation and a
stronger JEPA training study remain separate future protocols.
