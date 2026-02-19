# Experiment 2 — Pipeline Comparison Plan

## 1. Objective

Compare three end-to-end photo-to-inventory pipelines on the **same test set**
and **same metrics**, then evaluate robustness under controlled image degradations.

| Pipeline | Strategy                              |
|----------|---------------------------------------|
| A (VLM)  | Image → GPT-4o-mini → inventory       |
| B (YOLO) | Image → 14-class YOLO → inventory     |
| C (YOLO+CNN) | Image → objectness YOLO → crops → CNN → inventory |

---

## 2. Dataset Creation

### 2.1 Photography Guidelines

**Target:** 100–120 images, each containing 2–8 items from the 14 classes.

**Backgrounds — use realistic, varied surfaces:**
- Kitchen counter / wooden table / marble top
- Inside a fridge or shelf
- Grocery bag / shopping basket
- Chopping board or plate
- Tablecloth / placemat

**Do NOT use** a clean white or solid-color background.
The app operates in real environments; the test set must reflect that.

**Difficulty tiers (aim for roughly equal splits):**

| Tier | Images | Items/image | Description |
|------|--------|-------------|-------------|
| Simple   | ~35 | 2–3 | Distinct items, no occlusion, good lighting |
| Medium   | ~45 | 4–5 | Some similar items (lemon+orange), mild occlusion |
| Hard     | ~30 | 6–8 | Heavy occlusion, similar colors, mixed lighting |

**Shooting checklist:**
- Vary **angles**: top-down, 45°, slight side angle
- Vary **lighting**: natural daylight, warm indoor, dim/shadow
- Vary **spacing**: neatly arranged vs. piled/overlapping
- Include every class in at least 8–10 images across the full set
- Capture some **confusing pairs**: lemon/orange, peach/orange, tomato/apple,
  red pepper/tomato, potato/onion, cucumber/green pepper
- Use a phone camera (the same device the app targets)

### 2.2 Annotation

**Tool:** Roboflow (free tier), CVAT, or Label Studio.

**Format:** YOLO `.txt` (one file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalized [0, 1]. Class IDs per `config.py` (0=apple … 13=tomato).

**Export structure:**
```
dataset_exp2/
  images/       ← 100–120 .jpg files
  labels/       ← matching .txt annotation files
```

**Quality control:**
- Every object fully boxed, tight bounding box
- Green vs. red pepper annotated with correct class
- Partially occluded items still annotated
- No duplicate boxes on the same object

---

## 3. Image Degradation

Generate degraded versions of the **same** images to test pipeline robustness.
All degradations applied using `albumentations` / `PIL` / `OpenCV` with fixed
random seed for reproducibility.

### Degradation types

| ID | Degradation | Parameters | Simulates |
|----|-------------|------------|-----------|
| D1 | Gaussian blur | kernel=7, sigma=3.0 | Out-of-focus camera |
| D2 | Gaussian noise | mean=0, sigma=25 (uint8 scale) | Low-light sensor noise |
| D3 | JPEG compression | quality=15 | Messaging apps, cheap uploads |
| D4 | Downscale + upscale | factor=4× (e.g. 640→160→640) | Low-resolution camera |
| D5 | Brightness reduction | factor=0.4 (darken by 60%) | Poor / dim lighting |
| D6 | Motion blur | kernel=15, angle=random | Hand shake / movement |

Parameters above are starting points. Adjust after visual inspection so that
images are noticeably degraded but still recognizable by a human.

### Option A — Single combined degraded dataset

Apply **all** degradations simultaneously to each image (a "worst-case" combo):

```
dataset_exp2/
  images/           ← clean originals
  labels/           ← shared ground truth
  images_degraded/  ← all 6 degradations combined
```

**Evaluation runs:** 2 per pipeline = **6 total runs**.

| Run | Pipeline | Dataset |
|-----|----------|---------|
| 1   | A (VLM)  | clean   |
| 2   | A (VLM)  | degraded|
| 3   | B (YOLO) | clean   |
| 4   | B (YOLO) | degraded|
| 5   | C (YOLO+CNN) | clean   |
| 6   | C (YOLO+CNN) | degraded|

**Results table:**

|            | Clean F1 | Degraded F1 | F1 Drop (%) | Avg Latency |
|------------|----------|-------------|-------------|-------------|
| Pipeline A |          |             |             |             |
| Pipeline B |          |             |             |             |
| Pipeline C |          |             |             |             |

**Pros:** Simple, fast, one clear comparison.
**Cons:** Cannot tell which specific degradation hurts each pipeline.

---

### Option B — Separate degradation datasets

Generate **one folder per degradation type** so each can be evaluated independently:

```
dataset_exp2/
  images/              ← clean originals
  labels/              ← shared ground truth
  images_d1_blur/
  images_d2_noise/
  images_d3_jpeg/
  images_d4_lowres/
  images_d5_dark/
  images_d6_motion/
```

**Evaluation runs:** 7 per pipeline = **21 total runs**.

| Run | Pipeline | Dataset |
|-----|----------|---------|
| 1–3   | A / B / C | clean     |
| 4–6   | A / B / C | D1 blur   |
| 7–9   | A / B / C | D2 noise  |
| 10–12 | A / B / C | D3 jpeg   |
| 13–15 | A / B / C | D4 lowres |
| 16–18 | A / B / C | D5 dark   |
| 19–21 | A / B / C | D6 motion |

**Results table:**

|            | Clean | D1 Blur | D2 Noise | D3 JPEG | D4 LowRes | D5 Dark | D6 Motion | Avg Degraded |
|------------|-------|---------|----------|---------|-----------|---------|-----------|--------------|
| Pipeline A |       |         |          |         |           |         |           |              |
| Pipeline B |       |         |          |         |           |         |           |              |
| Pipeline C |       |         |          |         |           |         |           |              |

**Pros:** Rich analysis — shows exactly which degradation hurts which pipeline.
**Cons:** More runs (21 vs 6), more tables/charts. Note that Pipeline A (VLM)
costs ~$0.01–0.02/image, so 7×100 images ≈ $7–14 in API calls.

---

## 4. Metrics

All metrics are **count-based** (inventory-level), matching the existing
`evaluation/metrics.py` logic:

```
TP = min(predicted_count, ground_truth_count)
FP = max(0, predicted_count - ground_truth_count)
FN = max(0, ground_truth_count - predicted_count)
```

### 4.1 Primary Metrics (per pipeline, per dataset variant)

| Metric | Description |
|--------|-------------|
| **Micro Precision** | TP_total / (TP_total + FP_total) across all images and classes |
| **Micro Recall** | TP_total / (TP_total + FN_total) across all images and classes |
| **Micro F1** | Harmonic mean of micro precision and micro recall |
| **Macro F1** | Mean of per-class F1 scores (gives equal weight to rare classes) |

### 4.2 Per-Class Metrics

For each of the 14 classes:
- Precision, Recall, F1
- Total TP, FP, FN counts

Presented as a table + heatmap.

### 4.3 Latency

| Metric | Description |
|--------|-------------|
| **Mean latency** (ms/image) | Average inference time per image |
| **Median latency** (ms/image) | Robust to outliers |
| **Std deviation** (ms) | Consistency of speed |
| **P95 latency** (ms) | Worst-case tail |

For Pipeline C, also report **breakdown**: detection time + classification time.

### 4.4 Robustness Metrics (clean vs. degraded)

| Metric | Formula |
|--------|---------|
| **F1 Drop** | (F1_clean − F1_degraded) / F1_clean × 100% |
| **Robustness Score** | Mean F1 across all dataset variants (Option B) |
| **Worst-case F1** | Lowest F1 across all degradation types |

### 4.5 Error Analysis

| Error type | Definition |
|------------|------------|
| **Missed items** | FN breakdown: which classes are most often missed? |
| **Phantom items** | FP breakdown: which classes are hallucinated / over-counted? |
| **Over-counting** | pred > gt: how often and by how much? |
| **Under-counting** | pred < gt: how often and by how much? |
| **Confusion pairs** | Which classes get confused for each other? (confusion matrix) |

---

## 5. Execution Workflow

### Step 1 — Create dataset
1. Photograph 100–120 multi-item scenes (Section 2.1)
2. Annotate bounding boxes in YOLO format (Section 2.2)
3. Place in `dataset_exp2/images/` and `dataset_exp2/labels/`
4. Validate: run a script to check label/image pairing, class distribution

### Step 2 — Generate degraded datasets
1. Run degradation script(s) on `dataset_exp2/images/`
2. Output to `dataset_exp2/images_d1_blur/`, etc. (or `images_degraded/`)
3. Labels folder is shared — no changes needed

### Step 3 — Run pipelines
1. Run all 3 pipelines on the clean dataset
2. Run all 3 pipelines on each degraded dataset
3. Save predictions as JSON (per pipeline, per dataset variant)
4. Log latency per image

### Step 4 — Compute metrics
1. Compute primary metrics (Section 4.1) per pipeline per dataset variant
2. Compute per-class metrics (Section 4.2)
3. Compute latency stats (Section 4.3)
4. Compute robustness metrics (Section 4.4)
5. Run error analysis (Section 4.5)

### Step 5 — Generate report
1. Comparison tables (pipeline × dataset variant)
2. Bar charts: F1 per pipeline per condition
3. Heatmaps: per-class F1, confusion matrices
4. Radar/spider chart: pipeline strengths across conditions (optional)
5. LaTeX export for dissertation

---

## 6. Deliverables

| Artefact | Format |
|----------|--------|
| Clean test dataset | `dataset_exp2/images/` + `labels/` |
| Degraded datasets | `dataset_exp2/images_d*/` |
| Per-pipeline predictions | `results/exp2/{pipeline}_{dataset}_predictions.json` |
| Metrics summary | `results/exp2/comparison_summary.json` |
| Comparison bar charts | `results/exp2/comparison_*.png` |
| Confusion matrices | `results/exp2/{pipeline}_confusion.png` |
| LaTeX tables | `results/exp2/comparison_table.tex` |
| Experiment log | `logs/experiment2_{timestamp}.jsonl` |

---

## 7. Winner Selection Criteria

The **overall winner** is the pipeline with the best balance across:

1. **Clean F1** — baseline accuracy (most important)
2. **Robustness** — smallest F1 drop under degradation
3. **Latency** — speed for real-time app use
4. **Consistency** — low variance across classes and conditions

If no single pipeline dominates all criteria, present a weighted discussion
and justify the choice for Experiment 3 (app integration).
