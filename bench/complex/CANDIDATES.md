# Complex-paper benchmark candidates (Modal, < $5 GPU per paper)

Method: shallow `git clone --depth 1` of each repo (commit = HEAD on 2026-11-08), README/config/requirements inspection, GitHub REST issue search for "reproduce". No repo code was run, so every GPU-hour figure is an ESTIMATE from model size, data size and published times, not a measurement. Budget reference: A10G ~ $1.1/h (4-5 h), H100 ~ $3.3/h (1.5 h). Add ~15-40 min of image build and data download to every estimate.

Routes: (a) eval-only from released checkpoint, (b) full small config, (c) scaled-down run with relative claim.

Items marked "unverified" come from memory of the paper, not from files I opened.

---

## 1. LoRA - 2106.09685 "LoRA: Low-Rank Adaptation of Large Language Models"
- Repo: https://github.com/microsoft/LoRA @ c4593f0 (examples/NLU for RoBERTa/DeBERTa, examples/NLG for GPT-2)
- Claims: Table 2 (RoBERTa-base LoRA r=8 on MNLI/SST-2/MRPC/CoLA/QNLI/QQP/RTE/STS-B, ~87.2 avg), Table 3 (GPT-2 M/L E2E NLG: BLEU 70.4 etc.), plus LoRA-vs-FT/Adapter/BitFit rows.
- Routes:
  - (a) NLU: LoRA checkpoints are GitHub release assets (3.4 MB each for roberta-base; base weights from HF Hub). Eval of MNLI/SST-2/MRPC dev: ~0.1 h A10G. NLG: GPT-2 M LoRA checkpoints + `download_evalscript.sh` (E2E metric scripts). Beam-search decoding of E2E test: ~0.3 h A10G.
  - (b) RoBERTa-base LoRA fine-tuning on small GLUE tasks (RTE 2.5k, MRPC 3.7k, STS-B 5.7k, CoLA 8.5k; 30-80 epochs): ~0.2-0.5 h each on A10G; whole small-task set ~2 h. SST-2 (67k, 60 epochs) ~1.5 h. MNLI (393k, 30 epochs) is too long; skip or c.
  - (c) LoRA vs full FT on RTE/MRPC/STS-B: relative claim "LoRA matches FT with 0.3M params", ~1.5 h A10G.
- Data: GLUE via HF datasets/gdrive script (no login, ~100 MB). E2E dataset in repo scripts. GPT-2 M weights ~1.4 GB.
- Env risk: pins old transformers (examples/NLU installs a bundled transformers fork), torch 1.x-era, loralib is tiny. NLG uses fp16 + apex-style code, `torch.distributed.launch`; removed in modern torch (medium).
- Known issues (REST search): #127, #151, #165 "can't reproduce RoBERTa-base" (open): numbers 0.5-1 pt off, seed/hardware sensitivity. README itself warns results vary by GPU, CUDA, seed.
- Feasibility: HIGH (a and b on small tasks).

## 2. SimCSE - 2104.08821 "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
- Repo: https://github.com/princeton-nlp/SimCSE @ 102d450
- Claims: Table 5 (unsup SimCSE-BERT-base STS avg 76.25 on STS12-16, STS-B, SICK-R); Table 6 (sup, 81.57); ablation tables on dropout p, pooler, NLI data.
- Routes:
  - (a) HF Hub checkpoints (`princeton-nlp/unsup-simcse-bert-base-uncased`, sup variants). STS eval with SentEval: ~0.1 h on A10G, mostly dataset load.
  - (b) Unsup training: 1M wiki sentences, 1 epoch, bs 64, max len 32, ~15.6k steps. On a 3090 about 1-1.5 h; on A10G estimate ~1.5 h. Supervised (SNLI+MNLI, 275k pairs, 3 epochs) ~1.5-2 h A10G.
  - (c) Dropout-rate ablation (p=0.1 vs 0.0 vs 0.3): relative claim, 3 x 1.2 h with 0.5 epoch each -> ~2 h.
- Data: `data/download_wiki.sh` (~1M sentences, ~120 MB), `download_dataset.sh` for SentEval (~150 MB), NLI csv. No login.
- Env risk: requirements pin `transformers==4.2.1`, torch 1.7.1+cu110; faiss/Ampere warning in README; old pins fail with Python 3.11 (high drift).
- Known issues: #285 cannot reproduce `bert-base-uncased avg_first_last` numbers; #293 and #295 MLM-auxiliary-loss bugs (closed). STS-B dev is noisy; paper uses 3090s and says different hardware shifts results ~0.3-0.5.
- Feasibility: HIGH.

## 3. ELECTRA - 2003.10555 "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"
- Repo: https://github.com/google-research/electra @ 8a46635 (TensorFlow 1.15)
- Claims: Table 8 / README "expected results" GLUE dev for Small/Base; Table 1 relative: ELECTRA-Small vs BERT-Small-style; ablations Table 3 (replaced token detection vs MLM etc.), unverified.
- Routes:
  - (a) Released ELECTRA-Small checkpoint (GCS zip, ~50 MB) fine-tuned on GLUE dev (CoLA/SST/MRPC/RTE/STS-B/QNLI): each ~0.2-1 h on A10G (small model, 14M params; QNLI 105k x 3 epochs ~1 h). Total for the six small tasks ~3 h. ELECTRA-Base released too (110M): RTE/MRPC/STS-B/CoLA each ~0.3 h. Note "eval-only" here means fine-tune then eval, since the downloads are pretrained not fine-tuned (there is a `do_eval` path but fine-tuned checkpoints are not released).
  - (c) Pretrain small for 10-20k steps on a OpenWebText slice and compare RTD vs MLM objective. README: 200k steps = 10 h on V100, so 20k steps is ~1 h; GLUE after 20k steps is weak but the RTD>MLM ranking is a relative claim (risky at that scale, MEDIUM).
  - (b) Full 1M-step small pretraining = 4 days V100: out of budget.
- Data: GLUE (gist script, no login); OpenWebText 12 GB (HF/gdrive mirror, no login) plus 30 GB tfrecords for pretraining (disk heavy); vocab.txt on GCS.
- Env risk: TF 1.15 needs Python <=3.7 and CUDA 10; no Ampere/Hopper wheels (nvidia-tensorflow or CPU-only TF 1.15). Biggest dependency-drift item in the set. On A10G (sm_86) stock TF 1.15 is unusable; need NVIDIA's TF1 container (HIGH risk, is the point).
- Known issues: #73 electra_small_owt reproduced results lower; #93 WNLI trick; #94 checkpoint conversion to HF fails.
- Feasibility: MEDIUM (HIGH if image uses NGC TF1).

## 4. Transformer-XL - 1901.02860
- Repo: https://github.com/kimiyoung/transformer-xl @ 44781ed (tf/ and pytorch/)
- Claims: enwik8 base (12L, 41M) 1.06 bpc, WT103 base ppl 24.0, large 18.3 (Tables 1-3); ablation Table 6 (relative positional encoding, memory length).
- Routes:
  - (a) README provides pretrained TF checkpoints (enwik8, wt103, lm1b, text8) via `sota/*.sh` download. Eval of enwik8 (5M test chars) / WT103 test (245k tokens): TF eval of the large models needs ~12-16 GB, 0.3-1 h on A10G. The paper's SOTA models were trained on TPUs; the enwik8 SOTA model is 24L 277M. Base-size checkpoints may not be released (unverified), so confirm.
  - (b/c) PyTorch `run_enwik8_base.sh` (12L, 41M): the original run is 400k steps; on one A10G ~12-20 h. Scaled: 20-30k steps then compare with/without recurrence (mem_len 0 vs 128), ~1.5 h each (relative claim of Table 6 style). WT103 base: too slow.
- Data: `getdata.sh` pulls enwik8 (36 MB zip), WT103 (180 MB), text8. No login.
- Env risk: PyTorch README says "Pytorch 0.4", TF 1.x; codebase uses `nn.DataParallel`, removed-era APIs (`.data`, in-place on views) -> porting needed. High drift.
- Known issues: #47 wt103_large PyTorch prints 285M params vs 128M in paper; #120 TF memory; #8 questions.
- Feasibility: MEDIUM.

## 5. DETR - 2005.12872 "End-to-End Object Detection with Transformers"
- Repo: https://github.com/facebookresearch/detr @ 29901c5
- Claims: Table 1 (R50 42.0 AP, R50-DC5 43.3, R101 43.5, R101-DC5 44.9 at 500 epochs); Table 2 ablations (number of decoder layers, FFN, positional encodings, losses); panoptic Table 5.
- Routes:
  - (a) Released `.pth` (dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth ~160 MB; R101, DC5, panoptic variants). `main.py --eval` on val2017 (5k images): R50 ~10 min on A10G, R101-DC5 ~30 min. Whole Table 1 eval ~1.5 h A10G. Reproduces 42.0 AP (+/-0.1) in practice.
  - (c) Scaled training: 150-epoch/500-epoch impossible (R50 500 ep = 3 days on 8xV100 -> ~2000 GPU-h order). Scaled: train on 10-20% of COCO at low res for ~3 h; check relative claim e.g. aux-loss on vs off (Table 2). Weak, MEDIUM-LOW signal.
- Data: COCO 2017 from images.cocodataset.org (val 1 GB, annotations 241 MB; train 18 GB only if training). No login. Note README had a dead annotations link (#598); use cocodataset.org.
- Env risk: `pycocotools` build from git (needs cython, numpy ABI issues), `panopticapi` from git, `torch>=1.5` unpinned so modern torch works with small fixes. Low-medium risk. Good "easy-ish" detection representative.
- Known issues: #313 reproducibility (training variance), #451.
- Feasibility: HIGH for (a), LOW-MEDIUM for (c).

## 6. SegFormer - 2105.15203
- Repo: https://github.com/NVlabs/SegFormer @ 65fa8cf (mmsegmentation fork)
- Claims: Table 1 ADE20K (B0 37.4 mIoU ... B5 51.0), Cityscapes (B0 76.2 ... ), model-size ablations (Table 2), efficiency numbers.
- Routes:
  - (a) Weights on Google Drive / OneDrive folders (B0 ~15 MB, B1 ~55 MB, B2 ~100 MB, B5 ~340 MB). ADE20K val (2000 imgs, 512 short side) eval: B0-B2 ~5-10 min each on A10G; single-scale. Full B0-B5 sweep ~1 h. Drive-folder download via gdown can hit quota (risk). HF Hub has `nvidia/segformer-b*-finetuned-ade-512-512` mirrors (via transformers), a fallback for numbers, but not the mmseg code path.
  - (c) B0 training 160k iters bs16 on ADE20K: about 8 GPUs x ~5 h originally; on A10G >30 h. Scaled: 8k iters B0 (~1 h) then compare B0 vs B1 relative; weak.
- Data: ADE20K (SceneParse150, ~1 GB zip from MIT; no login); Cityscapes requires registration -> BLOCKER, skip. ImageNet-pretrained mit_b*.pth encoder weights from the same Drive folder.
- Env risk: README tested with CUDA 10.1, torch 1.7.1, `mmcv-full==1.2.7` (compiles CUDA ops; no wheel for modern torch/CUDA). Very high drift; the main value of this paper.
- Known issues: mostly feature/figure questions (#77, #123, #120); no major numeric mismatch found.
- Feasibility: MEDIUM (eval-only HIGH if mmcv build succeeds; drive quota is a risk).

## 7. DDIM - 2010.02502 "Denoising Diffusion Implicit Models"
- Repo: https://github.com/ermongroup/ddim @ e8df66c (PyTorch; "tested on PyTorch 1.6")
- Claims: Table 1: CIFAR-10 FID vs steps S in {10,20,50,100,1000} and eta in {0, 0.2, 0.5, 1, sigma-hat}; (eta=0, S=10: 13.36; S=100: 4.16, from memory, verify). Also CelebA 64.
- Routes:
  - (a) CIFAR-10/LSUN weights auto-downloaded from heibox.uni-heidelberg.de (`functions/ckpt_util.py`, the DDPM checkpoint converted by pesser, ~140 MB EMA). Command from README: `main.py --config cifar10.yml --use_pretrained --sample --fid --timesteps S --eta E`. 50k samples x S steps on a 35.7M-param UNet: S=10 ~10 min, S=100 ~1.5 h on A10G, S=20/50 in between. A table subset (S=10,20,50 x eta 0,1) ~2.5 h. Use 10k samples to be cheap but FID biases upward; state this.
  - (c) none needed.
- Data: CIFAR-10 (170 MB, auto) + FID reference stats; FID uses `pytorch-fid`/TF inception weights downloaded from GitHub/Google (no login). Inception FID implementation differences (TF vs pytorch-fid) shift numbers by ~0.1-0.5.
- Env risk: low-medium (old torch code, but pure PyTorch); heibox mirror availability is a single point of failure (note).
- Known issues: none surfaced by REST search (no matching issues). Community commentary: FID differences from different Inception implementations.
- Feasibility: HIGH.

## 8. Improved DDPM - 2102.09672
- Repo: https://github.com/openai/improved-diffusion @ 1bc7bbb
- Claims: Table 1/2 CIFAR-10 NLL and FID for L_simple vs L_hybrid vs L_vlb, linear vs cosine schedule; sampling-steps vs FID (Fig. 8) with 25/50/100/250/1000 steps.
- Routes:
  - (a) README checkpoints on openaipublic.blob.core.windows.net: `cifar10_uncond_50M_500K.pt` (L_hybrid, cosine) and `cifar10_uncond_vlb_50M_500K.pt` (L_vlb), ~200 MB each. Eval FID with 4 / 25 / 50 / 100 respaced steps on 10-50k samples: 50 steps x 50k ~1 h A10G. NLL eval via `image_nll.py` on CIFAR test: ~0.5 h.
  - (b/c) Train CIFAR-10 50M model: 500k steps bs128 is ~2-3 days on 8 V100s. Scaled: 30-50k steps (about 3 h A10G) compare L_hybrid vs L_simple FID/NLL ordering (small-scale relative claim; MEDIUM-LOW).
- Data: CIFAR-10 via `datasets/cifar_train.py` (170 MB); FID reference batch for CIFAR is NOT shipped (must be computed with TF inception, extra step, risk).
- Env risk: `mpi4py` (needs MPI libs) for even single-GPU runs, old torch/blobfile; tf-based FID evaluator lives in the separate guided-diffusion `evaluations` repo (TF 1.x). Medium-high.
- Known issues: none via REST search.
- Feasibility: MEDIUM-HIGH.

## 9. StyleGAN2-ADA (PyTorch) - 2006.06676 "Training GANs with Limited Data"
- Repo: https://github.com/NVlabs/stylegan2-ada-pytorch @ d72cc7d
- Claims: CIFAR-10 FID 2.92 (conditional, README says record FID 2.42 in paper table for TF), Fig. 11b CIFAR-10 FID vs data amount with ADA; AFHQ/METFACES FIDs in Table 2.
- Routes:
  - (a) `cifar10.pkl` and per-config pkls at nvlabs-fi-cdn.nvidia.com (~100-300 MB). `calc_metrics.py --metrics=fid50k_full` for CIFAR-10: ~15-25 min A10G (inception download from nvlabs CDN). AFHQ cat/dog/wild 512x512 FID ~1 h each (50k images at 512^2 on A10G; use fid50k with fewer if needed).
  - (c) Training from scratch on CIFAR-10 takes ~2 days on 8 V100 (unverified); scaled: ADA on 10% CIFAR-10 for ~1000 kimg (~3-4 h A10G) to check ADA beats no-ADA at small data (relative claim, weak at that duration, MEDIUM-LOW).
- Data: CIFAR-10 -> zip via `dataset_tool.py`; AFHQ 512 from the repo's link (~1.3 GB, no login).
- Env risk: custom CUDA ops (`upfirdn2d`, `bias_act`, `filtered_lrelu`-style) JIT-built by ninja at first use; README requires PyTorch 1.7.1, CUDA toolkit 11+, gcc/ninja; fails on modern torch/CUDA without patching (known). Pickled `.pkl` networks import repo's `torch_utils` module paths. HIGH drift (good test).
- Known issues: #283 FFHQ-70k reproducibility, #287 poor quality from official pkl (usually env-induced), #158.
- Feasibility: MEDIUM (a is HIGH once ops compile).

## 10. SimSiam - 2011.10566 "Exploring Simple Siamese Representation Learning"
- Repo (faithful widely-used PyTorch reproduction; official FB repo is ImageNet-only): https://github.com/PatrickHua/SimSiam @ 75a7c51. Official: https://github.com/facebookresearch/simsiam (ImageNet, blocked).
- Claims: paper CIFAR-10 results in Appendix (ResNet-18, kNN/linear acc ~91.8 at 800 ep); stop-gradient ablation (Table 1 of paper: without stop-grad collapses, acc 10%, with 91%), predictor ablation, batch-size ablation, BN ablation. These are checkable relative claims on CIFAR-10.
- Routes:
  - (b) README log: 800 epochs took 11 h 46 min on one GPU (their GPU, probably 2080Ti-class), final linear acc 90.83. Too long. A10G: ~50 s/epoch -> 100 epochs ~1.4 h; 200 epochs ~3 h.
  - (c) 100-200 epochs + linear eval (~0.2 h): relative ablations: stop-grad vs no stop-grad (collapse, expected acc ~10-20% vs ~80%), predictor identity vs MLP, SimSiam vs SimCLR/BYOL configs in `configs/` (simclr_cifar.yaml, byol_cifar.yaml present). Three 60-epoch runs for ~1.5 h total fits budget.
- Data: CIFAR-10 (auto, 170 MB). No login.
- Env risk: requirements pin torch==1.7.0, torchvision 0.8.1, Pillow 7.2, numpy 1.18.5, `yaml==5.3.1` (invalid pip name, should be pyyaml) -> install fails as written (dependency drift flag). Low runtime risk once fixed.
- Known issues: #47 logger bug. README numbers (90.83 at 800 ep vs paper 91.8) show the 1 pt gap.
- Feasibility: HIGH for (c).

## 11. TD3 - 1802.09477 "Addressing Function Approximation Error in Actor-Critic Methods"
- Repo: https://github.com/sfujim/TD3 @ 34770cc (official)
- Claims: Table 1 max average return over 1M steps on HalfCheetah-v1/Hopper/Walker2d/Ant/Reacher/InvPend/InvDoublePend (TD3 9636 HalfCheetah, 3564 Hopper, 4682 Walker, 4372 Ant; unverified). Table 2 / Fig: TD3 vs DDPG vs SAC/PPO/TRPO/ACKTR baselines. Ablation (Table 3): clipped double Q, target smoothing, delayed updates.
- Routes:
  - (b) Full 1M steps: small MLP (256-256) on one environment: CPU/GPU about 2.5-3.5 h on A10G (bounded by simulator + 1 update/step). 5 seeds is too much; 1 seed x 300k steps ~1 h per env (c).
  - (c) TD3 vs DDPG (`OurDDPG.py`, `DDPG.py` in repo) on HalfCheetah/Hopper at 300k steps: relative ranking TD3 > DDPG; ~1 h per run with CPU simulator; 4 runs ~4 h on A10G or ~$1.5 on CPU instances. CPU-only Modal containers would make this cheaper.
- Data: none (simulator).
- Env risk: README targets PyTorch 1.2, Python 3.7, `gym` MuJoCo `-v2` envs via `mujoco-py` (needs MuJoCo 2.0 binaries and old Cython; mujoco-py is dead; `-v2` removed from modern gymnasium, replaced by `-v4`/`v5` using `mujoco`). Reproduction changes the env version -> numbers shift. HIGH drift.
- Known issues: #19 Humanoid; #46 asks for plots/graphs. Results are seed-variance dominated.
- Feasibility: MEDIUM (cost fine; challenge is env pin and variance).

## 12. Decision Transformer - 2106.01345
- Repo: https://github.com/kzl/decision-transformer @ e2d82e6 (gym/ and atari/)
- Claims: Table 2 gym D4RL: DT on HalfCheetah/Hopper/Walker2d x medium/medium-replay/medium-expert (e.g. hopper-medium 67.6, walker-medium 74.0, halfcheetah-medium 42.6) vs CQL/BEAR/BRAC-v/AWR/BC; ablations (context length K, return conditioning, Table 4/5).
- Routes:
  - (b) `experiment.py --env hopper --dataset medium --model_type dt`: 3-layer GPT (128 hidden, ~0.8M params), 10 iterations x 10k steps = 100k gradient steps; each iteration includes 100 eval episodes... on A10G ~0.3-0.7 h per (env,dataset). Nine cells ~4-5 h, so do 4-6 cells (~2.5 h). Includes BC baseline (`--model_type bc`) in same script for relative check DT vs BC (~same cost).
  - (c) DT vs BC on medium-replay (where paper claims DT >> BC): relative claim, ~1.5 h for two envs.
- Data: D4RL via `download_d4rl_datasets.py` (needs `d4rl` + `mujoco-py`; data hosted at rail.eecs.berkeley.edu/datasets, pkl ~50-400 MB each; no login). Datasets can also be fetched through `minari`/HF mirrors.
- Env risk: `conda_env.yml`: python 3.8.5, gym 0.18.3, mujoco-py 2.0.2.13, torch 1.8.1, transformers 4.5.1, wandb 0.9.1, cudatoolkit 10; D4RL needs MuJoCo 2.0 binaries (HIGH).
- Known issues: #42 MuJoCo results depend strongly on env version (v2 vs v3); #48/#83 shortest-path code missing.
- Feasibility: MEDIUM-HIGH.

## 13. GraphSAINT - 1907.04931
- Repo: https://github.com/GraphSAINT/GraphSAINT @ c9b1e34 (TF + PyTorch + C++ cython samplers; `train_config/table2/*.yml` give exact configs per dataset/sampler)
- Claims: Table 2 F1-micro: PPI 0.966 (RW), Flickr 0.511, Reddit 0.966, Yelp 0.653, Amazon 0.815; ogbn-products 0.8027 (PyTorch, OGB). Table on samplers (node/edge/RW/MRW), deeper GNN configs in `train_config/explore`.
- Routes:
  - (b) PyTorch version, PPI/Flickr/Reddit with the shipped configs: PPI ~0.2 h, Flickr ~0.3 h, Reddit ~0.5 h, Yelp ~1 h, Amazon (1.6M nodes, 132M edges) ~2+ h with high RAM (>= 64 GB). A10G adequate for all but Amazon. ogbn-products (2.4M nodes, 62M edges) ~2-3 h, RAM >= 64 GB. Subset PPI+Flickr+Reddit+Yelp ~2.2 h.
  - (c) Sampler comparison (RW vs node vs edge) on Flickr/PPI: relative ranking in Table 2, ~1.5 h total.
- Data: Google Drive folder (author link) for PPI, Flickr, Reddit, Yelp, Amazon (~several GB; gdown quota risk) with Baidu alternative; ogbn-products via ogb (snap.stanford.edu, ~1.5 GB, no login) plus `ogbn_converter.py`.
- Env risk: requires cython-built sampler (`cython_sampler.pyx`), python>=3.6.8, tensorflow>=1.12 or pytorch>=1.1; README says paper results come from TF version while OGB results come from PyTorch (mixed provenance). Cython 3 and numpy ABI changes break the build (medium-high); optional C++ OpenMP sampler.
- Known issues: #19, #7 (Cluster-GCN F1 mismatch on Reddit). README warns inductive ogbn-products isn't comparable to leaderboard.
- Feasibility: MEDIUM-HIGH. Covers the "graph at larger scale" axis.

## 14. FT-Transformer / "Revisiting Deep Learning Models for Tabular Data" - 2106.11959
- Repo: https://github.com/yandex-research/rtdl-revisiting-models @ e3ed46c
- Claims: Table 3 (11 datasets: California Housing RMSE, Adult acc, Higgs, Covtype, Microsoft, Yahoo, Year, Epsilon, Helena, Jannis, Aloi): FT-Transformer vs ResNet vs MLP vs XGBoost/CatBoost, default and tuned; ablations in the appendix; hyperparameter spaces in appendix tables.
- Routes:
  - (a) Repo ships `output/` with tuned configs and results (per-dataset directories with trained model reports) - check whether weights or only metrics; if only metrics, route (b).
  - (b) Evaluate the final tuned configs for small datasets (California 20k rows, Adult 49k, Helena 65k, Jannis 83k, Aloi 108k): 15-line training runs ~1-10 min on A10G each x 15 seeds -> ~1 h per dataset total. Full HP tuning (Optuna 100 trials) is out for large sets; tune 20-30 trials on California/Adult as scaled (c): ~1.5 h.
  - (c) Relative claim: FT-T >= ResNet > MLP on most datasets; GBDT competitive on HIGGS/Covtype where FT-T leads, ~3 h across 5 datasets with fewer seeds.
- Data: single Dropbox tarball `revisiting_models_data.tar.gz` (~1-2 GB, no login; Dropbox link rot is a risk). Includes preprocessed datasets and splits.
- Env risk: conda env python 3.8.8, pytorch 1.7.1, cudatoolkit 10.1.243, cudnn 7.6.5 plus node.js for jupyter widgets; rtdl package in `package/`. Old env; moderate.
- Known issues: #28 "About reproducing the results" (closed). Seeds produce +/-0.01 variation.
- Feasibility: HIGH. Covers tabular axis cheaply.

## 15. Mamba - 2312.00752
- Repo: https://github.com/state-spaces/mamba @ e9594ce
- Claims: Table 3 zero-shot (lm-evaluation-harness: LAMBADA, HellaSwag, PIQA, ARC-E/C, WinoGrande, Pile ppl) for mamba-130m ... 2.8B; Fig. 8 efficiency (scan faster than FlashAttention-2 at long seq); Table 4 Selective-Copy and Induction Heads synthetic tasks (the latter trainable in minutes).
- Routes:
  - (a) HF Hub `state-spaces/mamba-130m|370m|790m|1.4b|2.8b` (0.5-11 GB). lm-eval zero-shot tasks for 130m/370m/790m: ~0.5-1.5 h on A10G; 1.4b ~1.5 h; 2.8b is memory-tight in fp16 (5.6 GB fine) ~2 h. LAMBADA and HellaSwag dominate runtime.
  - (c) Selective-copy synthetic task (Table 1 of paper): small 2-layer models trained for minutes; relative claim S6 > S4 on selective copy. Needs custom training code (not in repo; uses external `state-spaces/s4`), skip unless harness provides.
- Data: lm-eval datasets from HF (no login). Pile not needed for zero-shot.
- Env risk: CUDA extensions `selective_scan_cuda` and `causal-conv1d` compiled against specific torch/CUDA, `MAMBA_KEEP_CUDA_BUILD`/prebuilt wheel flow in README; nvcc needed (use `nvidia/cuda:*-devel` image); reproducing eval needs a specific lm-eval version. Medium-high.
- Known issues: #493 / #573 reproducibility of paper numbers (training); #268 DNA pretraining. Zero-shot numbers from HF checkpoints are reproducible within ~0.5 pt.
- Feasibility: HIGH for a (up to 790m), MEDIUM for including bigger models.

---

## Smoke papers (tiny, < $0.5, for testing the Modal backend)
| Paper | Repo @ commit | Task | Est. cost |
|---|---|---|---|
| GPT / nanoGPT char-level Shakespeare (GPT-2 arch; arXiv 2019 OpenAI, repo is Karpathy) | karpathy/nanoGPT @ 3adf61e | `prepare.py` (1 MB) + `train.py config/train_shakespeare_char.py`; README: best val loss ~1.47 in ~3 min on A100 | ~0.1 h A10G ~$0.15 |
| SAM - 2010.01412 (unofficial faithful, davda54) | davda54/sam @ 3c3afdb | WRN-16-8 on CIFAR-10; README: SGD 3.20 vs SAM 2.86 vs ASAM 2.55 % error at 200 epochs. Smoke: 10-20 epoch SGD vs SAM relative check; 200 epochs is 5+ h on A10G (SAM = 2x cost) so only short | ~0.3-0.5 h ~$0.5 |
| Grokking - 2201.02177 (teddykoker re-implementation) | teddykoker/grokking @ 34aef76 | modular-arithmetic transformer, minutes on GPU; checks grokking delay in val acc | ~0.2 h ~$0.25 |
| Lottery Ticket Hypothesis - 1803.03635 | facebookresearch/open_lth @ 2ce732f | LeNet-300-100 on MNIST iterative magnitude pruning, a few rounds; cheap relative claim (winning tickets beat random reinit) | ~0.3 h ~$0.3 |
| (optional) SimCSE eval-only (item 2a) | as above | tiny HF checkpoint STS eval | ~0.1 h |

(Smoke repos were cloned and listed, but their configs were not inspected beyond README; confirm before use.)

---

## Recommended set of 10 (maximizes diversity and feasibility)

| # | Paper | Domain | Route | Est. A10G h (~$) | Feasibility | Main risk |
|---|---|---|---|---|---|---|
| 1 | LoRA 2106.09685 | NLP efficient FT | a (+b small GLUE) | 2-3 (~$3) | High | seed variance (#127/#151/#165), old transformers fork |
| 2 | SimCSE 2104.08821 | NLP embeddings | b (unsup 1 epoch) + a | 2-3 (~$3) | High | transformers==4.2.1, faiss/Ampere |
| 3 | ELECTRA 2003.10555 | NLP pretrain/FT | a (fine-tune released small) | 3-4 (~$4) | Medium | TF 1.15 on Ampere (needs NGC TF1) |
| 4 | DETR 2005.12872 | Detection | a (Table 1 AP) | 1-1.5 (~$1.5) | High | pycocotools build; train claims not feasible |
| 5 | SegFormer 2105.15203 | Segmentation | a (ADE20K B0-B2) | 1-2 (~$2) | Medium | mmcv-full 1.2.7 build, Drive quota |
| 6 | DDIM 2010.02502 | Diffusion | a (Table 1 FID, subset) | 2-3 (~$3) | High | heibox mirror, FID impl. differences |
| 7 | SimSiam 2011.10566 | Self-supervised | c (stop-grad ablation, CIFAR-10) | 2-3 (~$3) | High | requirements.txt broken pins |
| 8 | Decision Transformer 2106.01345 | Offline RL | b (3-6 cells + BC) | 2-3 (~$3) | Medium-High | mujoco-py/D4RL, env version drift |
| 9 | GraphSAINT 1907.04931 | Graph, larger scale | b (PPI/Flickr/Reddit/Yelp, +products if RAM allows) | 2-3 (~$3) | Medium-High | cython sampler, Drive data, RAM |
| 10 | FT-Transformer 2106.11959 | Tabular | b/c | 2-3 (~$3) | High | Dropbox tarball, old torch/cuda pins |

Reserves, in order: Mamba (custom CUDA kernels, high-value env test), StyleGAN2-ADA (custom ops), Improved DDPM (relative L_hybrid vs L_simple), TD3 (RL, mujoco-py), Transformer-XL (port old PyTorch 0.4).

Not selected and why: MAE, DINO, ConvNeXt, MoCo, DeiT/Swin (ImageNet needs login, BLOCKER; the scaled alternatives on ImageNet-100 still need ImageNet), Stanford Alpaca / alpaca-lora / QLoRA (LLaMA weights gated, 7B training > budget), score_sde (JAX/TF 2.4 pins, 1000-step PC sampler: FID of 50k samples is ~10+ GPU-hours), DPO/RLHF (7B+, > budget), Cityscapes-based mmseg configs (registration wall).

Cross-cutting notes for the benchmark harness:
- Several checkpoint hosts are flaky or quota-limited (Google Drive folders: SegFormer, GraphSAINT; Dropbox: FT-Transformer; heibox: DDIM; Azure blob: Improved DDPM). Pre-stage mirrors in a Modal Volume and record the sha256 to keep runs inside budget.
- GitHub GraphQL is blocked in this sandbox (`gh search issues` fails); REST `gh api search/issues` works. Issue lists above are only top-3 hits per repo, not exhaustive.
- GPU-hour numbers need one calibration run each (an A10G is roughly 1/2 of an A100 for these workloads, similar to a 3090).
