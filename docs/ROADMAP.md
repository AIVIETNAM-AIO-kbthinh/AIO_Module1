# Roadmap & progress tracker

Tracks what is done and what is next. Phases follow the 4-week plan in
[research_plan_merged_project.md](../research_plan_merged_project.md). Roles:
Lead, Ops (AI pipeline), Model (AI model), Data (data engineer), QA.

Legend: `[x]` done · `[ ]` not started · `[~]` in progress.

## Phase 0 — Foundation harness (done)

- [x] Virtual environment, pinned `requirements.txt`, GPU env check script.
- [x] `src/utils`: seeding, typed config loader with validation, constants, logging.
- [x] `src/data`: MedMNIST loader, 3 preprocessing arms (P0/P_global/P_local),
      stratified subsampling by regime. Augmentation shared across arms and config-driven.
- [x] `src/models`: timm backbones (any ResNet/ViT id), LP/FT/LoRA arms with peft.
- [x] `src/evaluation`: AUROC/accuracy (MedMNIST-aligned), macro-F1, ECE.
- [x] `src/training`: train loop, best-by-val selection, runner, CLI entry.
- [x] CPU smoke tests for every module (run end to end on this machine).

## Next up

### 1. Baseline reproduction on GPU (Week 1 milestone) 
- [ ] Run ResNet-50 / P0 / 100% / FT on each dataset on the GPU server.
- [ ] Compare AUROC/accuracy against the published MedMNIST numbers (within tolerance).
- [ ] Record the baseline table; confirm the harness is trustworthy before scaling up.

### 2. Experiment-matrix orchestration (Week 2) 
- [x] Define the locked core matrix in one place (`configs/matrix/core.yaml`):
      3 preprocessing x 3 transfer x 3 regime {100%, 10%, 5%} x 2 datasets x 3 seeds = 162 runs.
- [x] Generate all 162 config YAMLs automatically with systematic names (`scripts/generate_matrix.py`).
- [x] Produce a slice-assignment sheet splitting runs across 5 Kaggle accounts round-robin (`assignment.csv`).
- [ ] (Optional) a batch runner that executes one slice and appends to the shared results CSV.

### 3. Run the core matrix (Weeks 2-3) 
- [ ] Batch 1: regimes 100% and 10%.
- [ ] Batch 2: regime 5% and the second dataset; complete all 162 runs.
- [ ] Keep checkpoints and `results.csv` consolidated.

### 4. QA (continuous) 
- [ ] Leakage checks (no train/test overlap), seed integrity, correct metrics.
- [ ] Reproduce 2-3 random runs and confirm the numbers match.

### 5. Analysis (Week 3) 
- [ ] Interaction plots (preprocessing x regime; preprocessing x strategy).
- [ ] Two-way ANOVA / interaction test.
- [ ] Pareto frontier: AUROC vs trainable parameters.
- [ ] Grad-CAM qualitative comparison across preprocessing arms.

### 6. Report and release (Week 4) 
- [ ] Final eval on the official test split; main results table (mean +/- std).
- [ ] Figures, report (Intro/Related/Method/Experiments/Results/Discussion), slides.
- [ ] Reproducible repository with a README to reproduce all results.

## UI dashboard (researcher-facing)

- [x] Phase 0 — Streamlit stack, `src/ui/` scaffold, `docs/UI.md`, `scripts/run_ui.py`.
- [x] Phase 1 — Data services over `results.csv` / `metrics.json`.
- [x] Phase 2 — Results table and run detail pages.
- [x] Phase 3 — Charts and decision guide.
- [x] Phase 4 — Run one experiment from the UI.
- [x] Phase 5 — Matrix batch orchestration.

## Notes for tracking
- The core matrix is locked: extensions (third dataset, regime 25%, ViT-S
  robustness, CLAHE param sweeps) go in a separate set, not the 162 core runs.
- Backbone is config-selectable, so running ViT alongside ResNet is a config
  change; doing it for the full core matrix doubles the run count, so treat it
  as a scoped decision when allocating compute.
