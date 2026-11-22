# Pre-registered Study Plan: Imaging Factors, Evaluation Leakage, and Open-Set Uncertainty in Wildlife Re-Identification with MiewID

Written: 22 November 2026 (before running any model or touching the test set).

## 1. Research Question

Modern animal re-identification models such as **MiewID-msv3** (EfficientNetV2-M trained with sub-center ArcFace across 64 wildlife species) report high retrieval accuracy on standard splits. In real field deployments — such as identifying free-living bovids from handheld or trail-camera photographs — performance depends strongly on imaging and deployment factors: unseen days, unseen camera angles, opposite flank viewpoints, low light / motion blur, and previously unseen individuals entering a herd (open-set queries).

Using two bovid datasets from the public **WildlifeReID-10k** benchmark (`wildlifedatasets/wildlifereid-10k` on Kaggle):
1. **MultiCamCows2024** (5,112 images, 90 individual cows, 7 chronological days `2023-08-14` to `2023-08-20`, 3 cameras `cam 1/2/3`, all 90 individuals observed in all 3 cameras).
2. **NyalaData** (1,942 wild outdoor bovid images, 237 individual nyalas, 236 observed from both `left` and `right` sides).

We benchmark **MiewID-msv3** (`conservationxlabs/miewid-msv3`, zero-shot + lightweight cosine-margin head adapted strictly on training days) alongside **MegaDescriptor-L-384** (`hf-hub:BVRA/MegaDescriptor-L-384`) as a second standard wildlife re-ID backbone.

**Pre-training overlap note (documented upfront):** `MegaDescriptor-L-384` (WACV 2024) was pre-trained on the 2023 WildlifeDatasets suite, which includes `NyalaData` (though `MultiCamCows2024` post-dates its training set). `MiewID-msv3` was trained on 64 Wildbook wildlife species and not on `MultiCamCows2024`. Therefore, `MultiCamCows2024` is our clean zero-shot primary benchmark for both models, while `NyalaData` serves specifically to measure the relative same-side vs. opposite-side viewpoint drop.

## 2. Locked Splits (Seed 42)

All splits are generated deterministically in `01_data_check.ipynb` before any embeddings are extracted:

- **MultiCamCows2024 (primary benchmark):**
  - The 90 cows are partitioned deterministically into **70 known individuals** (present in the reference gallery), **10 validation-unknown individuals** (held out from gallery, used only on validation days), and **10 test-unknown individuals** (held out from gallery, used only on locked test days).
  - **Gallery (reference database):** days `2023-08-14`, `2023-08-15`, `2023-08-16` (known cows only).
  - **Validation queries:** days `2023-08-17`, `2023-08-18` (known + 10 val-unknown cows). Used to inspect factor drops, select the best uncertainty/open-set score (`top1_sim` vs. `margin_id`), and tune the lightweight adaptation head.
  - **Locked Test queries (touched ONCE in `03_test_set.ipynb`):** days `2023-08-19`, `2023-08-20` (known + 10 test-unknown cows).
  - **Leakage comparator (random split):** on days `2023-08-14` to `2023-08-18` (known cows), a random gallery/query split matched to the exact query count of the cross-day validation split.

- **NyalaData (viewpoint factor test):**
  - Uses the official `WildlifeReID-10k` split (`train` as reference pool / validation inspection, `test` held out for `03_test_set.ipynb`). Within test individuals that have both `left` and `right` images, we compare **same-side retrieval** (`left -> left`, `right -> right`) against **opposite-side retrieval** (`left -> right`, `right -> left`).

## 3. Pre-registered Hypotheses & Pass/Fail Bars

Every quantitative test rule is fixed before seeing test-set results (95% bootstrap confidence intervals over 1,000 query resamples):

- **H1 (Split leakage & camera shift on MultiCamCows2024):**
  - **H1a (Day shift):** Zero-shot `MiewID-msv3` Rank-1 accuracy on unseen days is lower than on the matched random split by at least **3.0 percentage points** (lower 95% CI of drop > 0).
  - **H1b (Camera shift):** Zero-shot `MiewID-msv3` cross-camera Rank-1 accuracy (query camera excluded from gallery) is lower than same-camera Rank-1 accuracy by at least **5.0 percentage points**.

- **H2 (Viewpoint & field degradation factors):**
  - **H2a (Opposite flank on NyalaData):** On wild outdoor bovids (`NyalaData` test), opposite-side (`left <-> right`) Rank-1 accuracy drops by at least **20.0 percentage points** relative to same-side retrieval for `MiewID-msv3`.
  - **H2b (Field image degradations on MultiCamCows2024 test):** Under three controlled handheld-photography degradations applied to query images — **low light** (brightness factor `0.35`), **defocus/motion blur** (Gaussian blur radius `3.0`), and **occlusion/tight crop** (center crop `0.70`) — at least one factor causes a Rank-1 drop >= **5.0 percentage points**, and we report the full per-factor ranking.

- **H3 (Pre-registered failure flagging & open-set unknown-animal detection):**
  - On Validation (`02_val_and_factors.ipynb`), we compare two simple, training-free confidence scores computed from L2-normalized gallery cosine similarities:
    1. `top1_sim`: maximum cosine similarity to any gallery image.
    2. `id_margin`: `sim(best identity) - sim(second-best distinct identity)`.
  - Whichever score has higher validation AUROC is locked before running `03_test_set.ipynb`.
  - **H3a (Closed-set error flagging):** On locked test queries (known cows), the locked score flags misidentified queries with **AUROC >= 0.75**.
  - **H3b (Open-set unknown-animal detection):** Distinguishing the 10 never-seen test cows from the 70 known cows on locked test days achieves **AUROC >= 0.80**.

- **H4 (Lightweight adaptation on training days only):**
  - Training a lightweight residual projection head (`Linear -> LayerNorm -> GELU -> Linear + residual`, L2-normalized, cosine ArcFace / SupCon loss) on top of frozen `MiewID-msv3` embeddings using **only Gallery days (`14–16`, 70 known cows)** improves cross-camera Rank-1 on locked test days (`19–20`) by at least **+2.0 percentage points** without dropping open-set AUROC by more than `0.03`.

## 4. Metrics & Failure Analysis

- Closed-set retrieval: **Rank-1**, **Rank-5**, and **mAP** (mean Average Precision over all gallery matches of the true identity), reported with 1,000x query-bootstrap 95% CIs.
- Open-set & error flagging: **AUROC**, **TPR at 5% FPR**, and **risk-coverage curve** (Rank-1 accuracy when abstaining on the lowest-confidence 10% / 20% of queries).
- Visual failure inspection: export contact sheets of the worst closed-set misidentifications and lowest-margin open-set cases so failure causes (lighting, blur, pose, background clutter) can be inspected directly.
