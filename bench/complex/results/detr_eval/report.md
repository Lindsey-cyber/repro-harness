# Reproduction report

- objective: Reproduce the COCO val2017 box AP of DETR with a ResNet-50 backbone (Table 1, 500-epoch model) by evaluating the released checkpoint with this repository.
- source: /tmp/claude-0/complex/detr

## Claims

| claim | paper | status | mean | runs | source version |
|---|---|---|---|---|---|
| detr_r50_coco_val2017_box_ap (COCO val2017 AP) | 0.42 | **reproduced** | 0.4200 | 0.4200 | v0 |

## Root causes

- `env.missing_dependency` [reported by agent] (repro.env): python -c "import pycocotools" -> ModuleNotFoundError: No module named 'pycocotools'; scipy also absent. The repo only documents conda installs / `pip install -U 'git+https://github.com/cocodataset/co
- `env.path_or_config` [reported by agent] (repro.env): main.py builds the *train* dataset unconditionally, even with --eval: `dataset_train = build_dataset(image_set='train', args=args)` -> FileNotFoundError: /cache/minicoco/annotations/instances_train201
- `env.path_or_config` [reported by agent] (repro.env): models/backbone.py instantiates torchvision resnet50 with pretrained=True, so an ImageNet weight download from download.pytorch.org happens at model construction. Pre-cached to /cache/torch/hub/checkp

## Changes

None.

## Plan
```json
[]
```
