# Paper update guide (2026-05-31)

This document records **everything added since the first v1 draft**, where the **numeric results** live, **what LaTeX still needs**, and how we **respond to external critic reviews**. Use it as the single checklist before arXiv upload.

**VM status:** `locateanythingsam-vm` is **TERMINATED** (all GPU work synced locally; safe to leave off).

---

## 1. What was added (experiments + assets)

### 1.1 Full validation benchmarks (done earlier)

| Split | Methods | n | JSON table |
|-------|---------|---|------------|
| RefCOCO val | hybrid, fast, DINO-Tiny, DINO-Base, GT-oracle | 3,811 | `../outputs/full_val/full_val_table.json` |
| RefCOCO+ val | hybrid, fast, DINO-Tiny, DINO-Base, GT-oracle | 3,805 | `../outputs/refcoco_plus/refcoco_plus_table.json` |
| RefCOCO-g val (Google) | hybrid, fast, DINO-Tiny, DINO-Base, GT-oracle | 5,000 | `../outputs/refcocog/refcocog_table.json` |

### 1.2 Test splits (new — VM run 2026-05-31)

| Run | Split | n | mIoU | P@0.5 |
|-----|-------|---|------|-------|
| **Locate-SAM2 hybrid** | RefCOCO testA | 1,975 | **0.807** | 93.1% |
| **Locate-SAM2 hybrid** | RefCOCO testB | 1,810 | **0.730** | 81.5% |
| **Locate-SAM2 hybrid** | RefCOCO+ testA | 1,975 | **0.766** | 88.2% |
| **Locate-SAM2 hybrid** | RefCOCO+ testB | 1,798 | **0.650** | 72.4% |
| DINO-Base + SAM2 | RefCOCO testA | 1,975 | 0.761 | 87.4% |
| DINO-Base + SAM2 | RefCOCO testB | 1,810 | 0.661 | 73.5% |
| DINO-Base + SAM2 | RefCOCO+ testA | 1,975 | 0.708 | 81.3% |
| DINO-Base + SAM2 | RefCOCO+ testB | 1,798 | 0.517 | 56.8% |

**Source files:** `../outputs/test_splits/*_summary.json`, consolidated `../outputs/test_splits/test_splits_table.json` (if generated).

**Claim:** Hybrid beats DINO-Base on **all four** test splits (same as val).

### 1.3 Analysis probes (new)

| Analysis | File | Key numbers |
|----------|------|-------------|
| Hybrid vs fast (500 RefCOCO val refs) | `../outputs/analysis/hybrid_fallback_stats.json` | Answer differs **10.8%**; box differs **5.4%**; neither mode empty-box on this sample |
| Negative-prompt / hallucination (8 images × 2 prompts) | `../outputs/analysis/hallucination_probe.json` | Box+mask emitted on **62.5%** of nonsense prompts; mean SAM score **0.90** when mask emitted |

**Regenerate:**

```bash
python scripts/probe_hybrid_fallback.py --subset-size 500
python scripts/probe_hallucination.py --n-images 8
```

### 1.4 Figures added under `figures/` (git-tracked)

#### Original qual panels (4 cases)

| Folder | Role |
|--------|------|
| `win_ref5466/`, `win_ref2764/` | Wins vs DINO-Tiny |
| `fail_ref2885/` | Wrong instance — ours fails |
| `both_ref3281/` | Both succeed |
| `comparison_wins.png`, `comparison_failures.png` | Grid summaries |

#### New failure taxonomy export (8 cases, 2026-05-31)

Manifest: `figures/failure_export_manifest.json`

| Folder | Failure mode | Example query | Notes |
|--------|--------------|---------------|-------|
| `fail_wrong_instance_ref20398/` | (i) wrong instance | (see `query.txt`) | Locate-SAM2-specific fail |
| `fail_wrong_instance_ref13650/` | (i) wrong instance | | |
| `fail_spatial_ref24664/` | (ii) spatial / ordinal | | e.g. left/right language |
| `fail_spatial_ref5750/` | (ii) spatial | ``zebra on left'' | ours 0.28 mIoU, DINO 0.94 |
| `fail_attribute_ref4832/` | (iii) attribute | | color/size ambiguity |
| `fail_attribute_ref18360/` | (iii) attribute | ``red bike'' | |
| `fail_rare_or_long_ref36776/` | (iv) rare / long | ``mtf member whole pizza'' | |
| `fail_rare_or_long_ref32628/` | (iv) rare / long | ``person in pink except the hat'' | |

Each folder: `image_raw.jpg`, `ours_overlay.png`, `dino_overlay.png`, `query.txt`, `metadata.json` (includes `label_for_paper`).

**Regenerate:**

```bash
python scripts/export_paper_figures.py \
  --ours-records outputs/full_val/locate_sam2_hybrid_full_records.json \
  --dino-records outputs/full_val/dino_tiny_full_records.json \
  --per-mode 2
```

#### Hallucination probe figures

`figures/hallucination_probe/` — 16 cases (8 images × 2 negative prompts).  
Use one vivid false-positive for the paper (e.g. `img510591_neg8763`: nonsense prompt → confident box on toilet scene).

---

## 2. Where results live (full map)

```text
locateanythingsam/
├── outputs/
│   ├── full_val/              RefCOCO val — main Table 1 source
│   ├── refcoco_plus/          RefCOCO+ val
│   ├── refcocog/              RefCOCO-g val
│   ├── test_splits/           NEW — testA/testB summaries + records
│   ├── analysis/              NEW — hybrid fallback + hallucination JSON
│   ├── ablations/             200-ref adapter table
│   ├── dino_sweep/            DINO threshold sweep
│   └── full_val/figures/      18-case qual export (gitignored; more cases)
│
├── research_paper/
│   ├── figures/               Bundled PDF assets (THIS FOLDER — git tracked)
│   ├── tables.tex             Main numeric tables (needs test table — see §3)
│   ├── qual_figures.tex       Qual LaTeX (needs expansion — see §3)
│   └── main.tex                 Narrative (partially updated)
│
└── experiments/
    ├── RESULTS.md             All val numbers in one place
    └── HALLUCINATION_NOTES.md Mitigation ideas for Discussion
```

**Logs (VM, copied locally if needed):**

- `~/missing_experiments.log` — RefCOCO-g + RefCOCO+ extras
- `~/paper_pack_nohup.log` — test splits + failed-then-fixed export
- `~/paper_pack_remainder.log` — export + probes (completed)

---

## 3. LaTeX changes still to make

These are **not all done in `.tex` yet** — the data and figures exist; the paper source needs wiring.

### 3.1 Tables (`tables.tex`)

| Change | Status | Action |
|--------|--------|--------|
| RefCOCO-g table (`tab:refcocog`) | **Done** in `tables.tex` | Verify numbers match `refcocog_table.json` |
| RefCOCO+ fast + oracle rows | **Done** | Verify |
| **NEW: testA/testB table** | **Not in LaTeX** | Add `tab:test-splits` ( appendix or main) with 8 rows above; emphasize hybrid vs DINO-Base |
| Lead with DINO-Base in caption text | **Partial** | Caption: “primary baseline = DINO-Base”; Tiny = efficiency reference |

**Suggested placement:** Appendix table “RefCOCO / RefCOCO+ test splits” OR second row block under main results in §5.1.

### 3.2 Qualitative figures (`qual_figures.tex`)

| Change | Status | Action |
|--------|--------|--------|
| Original 4 panels | **In LaTeX** | Keep |
| Failure modes (ii)–(iv) | **Figures exist, not in LaTeX** | Add Fig.~\ref{fig:qual-failures-taxonomy} with 2×2 grid from `fail_spatial_*`, `fail_attribute_*`, `fail_rare_or_long_*` |
| Hallucination example | **Figures exist, not in LaTeX** | Add Fig.~\ref{fig:hallucination} from `hallucination_probe/` |
| Use **Locate-SAM2 failures** not only DINO wins | **Partial** | New `fail_wrong_instance_*` and spatial cases where **ours** mIoU < DINO |

**Example LaTeX row (spatial failure):**

```latex
\qualpanelrow{figures/fail_spatial_ref5750/image_raw.jpg}{...dino...}{...ours...}
{Input}{}{DINO-Tiny + SAM2}{mIoU = 0.94}{Locate-SAM2}{mIoU = 0.28}
```

Read mIoU from each folder’s `metadata.json`.

### 3.3 Methods (`main.tex` §3)

| Topic | Include where | Text to add |
|-------|---------------|-------------|
| **`best_score` definition** | §3 Prompt-to-Mask Adapter | “When SAM2 returns multiple mask candidates (`multimask_output=true`), we select the candidate with highest **SAM predicted mask IoU** (`outputs.iou_scores` from SAM2.1).” |
| **`crop` + 5% padding** | §3 | One sentence: expand box by 5% before SAM crop to include context |
| **DINO thresholds** | §4 Experimental setup | 0.25/0.25 after 200-ref sweep (`dino_threshold_sweep.json`) |
| **Hardware caveat** | §4 | “Latency measured on NVIDIA RTX PRO 6000 Blackwell; **peak VRAM** (8.9 GB hybrid vs 3.1 GB DINO-Base) is the practical deployment constraint.” |
| **No supervised RES** | §4 or §2 Related | “We do not compare to LAVT, CRIS, UniRES++ etc. — those train on RES masks; our regime is frozen zero-shot modular pipelines.” |

### 3.4 Discussion / Limitations (`main.tex` §6)

| Topic | Include |
|-------|---------|
| Hybrid vs fast | “On full val, hybrid exceeds fast by only +0.003 mIoU (RefCOCO); hybrid answers differ on **10.8%** of a 500-ref sample — fallback helps long/spatial/rare refs, not typical COCO photos.” |
| Hallucination / false positives | “On unrelated prompts, LocateAnything still emits a box+mask **62.5%** of the time (small probe, n=16). Unlike DINO, there is no native detection threshold. Mitigations: reject unparseable boxes; optional SAM score gate (τ≈0.5); future work: uncertainty flag or DINO ensemble.” |
| OOD | Still **not run** — keep `ood_protocol.tex` as protocol-only OR label “future work” |
| LocateAnything training overlap | **Desk check pending** — read HF model card; if RefCOCO/COCO overlap unknown, one honest caveat sentence |

### 3.5 Abstract

Already updated for three val splits + DINO-Base deltas. **Add one line** after test results land in paper:

> “On RefCOCO testA/testB, hybrid reaches 0.807/0.730 mIoU vs 0.761/0.661 for DINO-Base.”

---

## 4. Critic review → response map

External review raised paper-killers and architectural critiques. Below: **verdict**, **what we did**, **where to address in paper**, **what we deliberately skip**.

### 🔴 Critical items

| Criticism | Valid? | What we did | Where in paper |
|-----------|--------|-------------|----------------|
| **Missing testA/testB** | Yes | Ran 8 test evals (hybrid + DINO-Base × 2 datasets × 2 splits) | New table §5 or appendix; cite `outputs/test_splits/` |
| **DINO-Tiny strawman** | Partly | Have DINO-Base on all splits; Tiny is Grounded-SAM literature baseline | Abstract + §5.1: **lead with +5.5…+10.5 vs Base**; footnote Tiny as low-compute reference |
| **Blackwell GPU / latency** | Partly | VRAM reported; latency is hardware-specific | §4 setup + Limitations: emphasize **8.9 vs 3.1 GB**; ms labeled “on our hardware” |
| **Qual cherry-picking / hidden failures** | Yes | 8 labeled failure exports + existing fail_ref2885 | Expand `qual_figures.tex`; §5.5 taxonomy **with figures** for modes (i)–(iv) |
| **Val-only incomplete** | Addressed | Val + testA/B + RefCOCO-g val | §4: “We report val and test splits for RefCOCO family; RefCOCO-g test omitted (less standard).” |

### 🟡 Architectural / scientific

| Criticism | Valid? | What we did | Where in paper |
|-----------|--------|-------------|----------------|
| **200-ref ablation too small** | Partly | Full-val fast vs hybrid already run; point-only collapse large | §5.3: “Ablations on 200-ref subset (hash …); directional only; full-val confirms fast≈hybrid.” |
| **Fast vs hybrid margin tiny** | Yes | Measured fallback rate 10.8% answer diff | §6 Discussion paragraph (see §3.4) |
| **`best_score` undefined** | Yes | Code uses SAM `iou_scores` | §3 Methods one sentence |
| **Hallucination on random prompts** | Yes (new) | Probe: 62.5% false positive rate | §6 Limitations + optional small figure |
| **No Grounded-SAM v1 (SAM1)** | Fair | Different segmenter breaks controlled claim | §2 Related work acknowledgment only — **do not run** |
| **No supervised RES table** | Fair | N/A by design | §2/§4 disclaimer sentence |
| **LocateAnything training data overlap** | Unknown | **TODO:** read HF card | §4 one caveat if overlap possible |
| **OOD / domain transfer** | Fair | Not run (needs custom images) | Keep `ood_protocol.tex` as protocol; v2 differentiator |

### ✍️ Writing style (“AI tell”)

| Issue | Action | Where |
|-------|--------|-------|
| “The result is not that… The result is that…” | Rewrite defensively structured sentences | §5.1, §6 |
| Methods reads like YAML | Prose-ify adapter rationale | §3 |
| Weak intro hook | Sharpen motivation (LocateAnything timing, modular RES gap) | §1 |
| Thin Discussion | Add interpretation: oracle gap, hybrid margin, hallucination, deployment VRAM | §6 |

**No experiment required** — editorial pass only.

---

## 5. Recommended paper structure after updates

```text
§5 Results
  5.1 Main quantitative (val + mention test in text)
  5.2 Oracle diagnostic (unchanged — strongest result)
  5.3 Adapter ablations (200-ref, flagged)
  5.4 Qualitative wins (existing Fig 3)
  5.5 Failure taxonomy (NEW Fig — modes i–iv + hallucination)
  Appendix A: testA/testB full table
  Appendix B: hybrid fallback + hallucination probe stats
```

---

## 6. Scripts reference (reproducibility)

| Script | Purpose |
|--------|---------|
| `scripts/run_test_splits.sh` | testA/testB batch (8 runs) |
| `scripts/run_paper_pack.sh` | test + export + probes |
| `scripts/export_paper_figures.py` | Labeled failures → `research_paper/figures/` |
| `scripts/probe_hybrid_fallback.py` | fast vs hybrid stats |
| `scripts/probe_hallucination.py` | Negative-prompt probe |
| `scripts/build_experiment_tables.py` | Consolidate JSON tables |

---

## 7. Still not done (explicit non-goals for this upload)

- [ ] OOD stress test with 50–150 custom domain images (`experiments/ood/`)
- [ ] RefCOCO-g **test** split eval
- [ ] Full-val adapter ablations
- [ ] SAM 2.1 small segmenter swap
- [ ] Grounded-SAM v1 / SAM1 baseline
- [ ] LaTeX wiring for test table + new qual figures (data ready, `.tex` pending)
- [ ] Author block / GitHub URL (`authors.tex`)
- [ ] LocateAnything training-data overlap sentence (HF card review)
- [ ] Writing polish pass (intro, discussion, methods prose)

---

## 8. Pre-upload checklist

1. [ ] Add test-split table to `tables.tex` or appendix
2. [ ] Expand `qual_figures.tex` with failure modes + one hallucination panel
3. [ ] Add `best_score`, VRAM, supervised-RES disclaimer to `main.tex`
4. [ ] Discussion: hybrid 10.8% fallback, hallucination 62.5% probe
5. [ ] Abstract: testA/B numbers + DINO-Base-first framing
6. [ ] `make` in `research_paper/` — verify all new figure paths resolve
7. [ ] VM **TERMINATED** ✓ (confirmed 2026-05-31)

---

## 9. Quick number reference (copy-paste for tables)

### Val (primary)

| Dataset | Hybrid mIoU | DINO-Base mIoU | Δ | Oracle mIoU |
|---------|-------------|----------------|---|-------------|
| RefCOCO | 0.772 | 0.717 | +5.5 | 0.836 |
| RefCOCO+ | 0.717 | 0.612 | +10.5 | 0.836 |
| RefCOCO-g | 0.746 | 0.666 | +8.0 | 0.815 |

### Test (new)

| Dataset | Split | Hybrid | DINO-Base | Δ |
|---------|-------|--------|-----------|---|
| RefCOCO | testA | 0.807 | 0.761 | +4.6 |
| RefCOCO | testB | 0.730 | 0.661 | +6.9 |
| RefCOCO+ | testA | 0.766 | 0.708 | +5.8 |
| RefCOCO+ | testB | 0.650 | 0.517 | +13.3 |

---

*Last updated: 2026-05-31 after VM paper pack completed and synced locally.*
