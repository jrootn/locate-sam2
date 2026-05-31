# Locate-SAM2 research paper (v1 draft)

LaTeX source for the **systems + evaluation** paper on zero-shot referring-expression segmentation. This folder is self-contained for writing and PDF builds; the implementation and eval code live in the parent repo (`locate_sam2/`, `scripts/`).

**Scope:** research pipeline and benchmarks only — not a plugin or commercial app.

> **Latest experiment pack (2026-05-31):** test splits, failure figures, hallucination probe — see **[UPDATE_GUIDE.md](UPDATE_GUIDE.md)** for full results map, critic responses, and LaTeX TODO checklist. VM is **stopped**; all artifacts synced locally.

---

## Folder structure

```text
research_paper/
├── README.md           ← you are here (structure, tools, workflow)
├── STATUS.md           ← v1 checklist: done / in progress / todo
├── Makefile            ← build PDF with Tectonic
├── main.tex            ← paper source (abstract → conclusion)
├── references.bib      ← BibTeX citations
├── authors.tex         ← edit name / affiliation / email (title page)
├── ood_protocol.tex    ← OOD stress test section (in main PDF)
├── ood/                ← CSV templates + scoring workflow
├── figures/            ← bundled images for the PDF (tracked in git)
│   ├── win_ref5466/    ← qualitative: large win vs DINO-Tiny
│   ├── fail_ref2885/   ← qualitative: grounding failure
│   └── both_ref3281/   ← qualitative: both methods succeed
└── build/              ← generated PDF + logs (gitignored)
    └── main.pdf        ← output after `make`
```

### What each figure folder contains

| Folder | RefCOCO ref | Expression | Role in paper |
|--------|-------------|------------|----------------|
| `win_ref5466/` | 5466 | “right white spoon” | DINO mIoU 0.00 → ours 0.98 |
| `fail_ref2885/` | 2885 | “man turned around” | Ours 0.00 → DINO 0.84 (wrong grounding) |
| `both_ref3281/` | 3281 | “biblia sacra vulgata book” | Both ~0.98 mIoU |

Per-folder files (same layout everywhere):

- `image_raw.jpg` — COCO input image  
- `dino_overlay.png` — GroundingDINO-Tiny + SAM2 mask overlay  
- `ours_overlay.png` — Locate-SAM2 hybrid overlay  

Original exports (with `metadata.json`, more cases) live under `outputs/full_val/figures/` in the parent repo (gitignored).

---

## Tools used

### PDF engine: Tectonic (required)

This project does **not** use `pdflatex` or `latexmk`. Builds use **Tectonic 0.15.0** (XeTeX-based), same as the CV templates in `application_system`:

| Item | Path |
|------|------|
| Default binary | `/home/jroot/projects directory/application_system/tools/tectonic` |
| Alternate | `/home/jroot/projects directory/bin/tectonic` |
| Package cache | `~/.cache/Tectonic` (~54 MB bundle, auto-downloaded on first build) |

Tectonic resolves LaTeX packages from the bundle; no system TeX Live install is required.

### LaTeX stack in `main.tex`

- Document: `article` (11pt, A4), arXiv-style single column  
- Packages: `geometry`, `graphicx`, `booktabs`, `natbib`, `hyperref`, `tikz`, `subcaption`  
- Bibliography: `plainnat` + `references.bib`  
- Figures: JPEG/PNG under `figures/` (relative paths)

### Models and code (parent repo — for reproducibility)

| Component | Model / artifact |
|-----------|------------------|
| Grounder (ours) | `nvidia/LocateAnything-3B` |
| Grounder (baseline) | `IDEA-Research/grounding-dino-tiny` |
| Segmenter | `facebook/sam2.1-hiera-large` |
| Eval data | RefCOCO val, 3,811 refs, COCO `train2014` |

---

## LaTeX structure (v2 — fixed section order)

| File | Role |
|------|------|
| `main.tex` | Full paper; sections 1–6 + reproducibility + references |
| `preamble.tex` | Packages, `[H]` floats, qual helpers |
| `results_figure.tex` | One 2×2 results panel + ablation chart |
| `qual_figures.tex` | Qualitative figure* grids |
| `ood_protocol.tex` | §4.6 OOD (fixed `[H]` tables) |
| `authors.tex` | Title author block |
| `GAPS.md` | What is still missing for submission |

**Section order in PDF:** Intro → Related Work → Method → Experiments (4.1–4.6) → Discussion → Conclusion → Reproducibility → **References last** (`\clearpage` before bibliography).

Qualitative rows: **(a–d) title + expression**, then Input / DINO / Ours with mIoU under each panel.

## Build the PDF

```bash
cd research_paper
make
```

Output: **`build/main.pdf`** (~9 MB with embedded figure images).

```bash
make clean          # remove build/
make TECTONIC="/path/to/tectonic"   # override binary
```

Manual build (debug):

```bash
"/home/jroot/projects directory/application_system/tools/tectonic" \
  --print --keep-logs --outdir build main.tex
```

Offline (if cache is warm):

```bash
tectonic --only-cached --outdir build main.tex
```

---

## Numbers in the paper (source of truth)

All quantitative claims in `main.tex` come from the parent repo eval artifacts:

| Result | File (parent repo) |
|--------|---------------------|
| Main table (full val) | `outputs/full_val/full_val_table.json` |
| Ablations (200 refs, seed 42) | `outputs/ablations/ablation_table.json` |
| DINO threshold sweep | `outputs/dino_sweep/dino_threshold_sweep.json` |
| Qualitative metadata | `outputs/full_val/figures/*/metadata.json` |

**RefCOCO val (3,811 refs):**

| Method | mIoU | P@0.5 | Latency |
|--------|------|-------|---------|
| DINO-Tiny + SAM2 | 0.441 | 48.6% | 146 ms |
| DINO-Base + SAM2 | 0.717 | 81.7% | 164 ms |
| Locate-SAM2 fast | 0.769 | 87.5% | 190 ms |
| Locate-SAM2 hybrid | **0.772** | **88.1%** | 195 ms |
| GT-box + SAM2 oracle | 0.836 | 95.2% | 72 ms |

**RefCOCO+ val (3,805 refs):**

| Method | mIoU | P@0.5 |
|--------|------|-------|
| DINO-Tiny + SAM2 | 0.440 | 47.7% |
| DINO-Base + SAM2 | 0.612 | 69.4% |
| Locate-SAM2 hybrid | **0.717** | **81.6%** |

Oracle-relative (RefCOCO): **92.3%**. Gains vs DINO-Base: **+5.5** mIoU (RefCOCO), **+10.5** (RefCOCO+).

When eval reruns, update `main.tex` tables from the JSON files above, then `make`.

---

## Refresh figures

1. Re-export from parent repo (needs data + models):

   ```bash
   python scripts/export_qualitative.py \
     --ours-records outputs/full_val/locate_sam2_hybrid_full_records.json \
     --dino-records outputs/full_val/dino_tiny_full_records.json
   ```

2. Copy chosen cases into `research_paper/figures/<name>/` (three files each: `image_raw.jpg`, `dino_overlay.png`, `ours_overlay.png`).

3. Update captions in `main.tex` using each case’s `metadata.json` (ref id, query, mIoU).

4. `make`

---

## Git ignore

Generated artifacts are ignored at repo root:

- `research_paper/build/`
- `research_paper/*.pdf`, `*.aux`, `*.log`, etc.

**Tracked:** `main.tex`, `references.bib`, `figures/`, `Makefile`, this README, `STATUS.md`.

---

## Related docs (parent repo)

| File | Purpose |
|------|---------|
| `PAPER.md` | Markdown technical outline |
| `README.md` | Pipeline quick start and results summary |
| `outputs/experiments/EXPERIMENT_LOG.md` | Run history |

---

## Authors

Edit `authors.tex` before submission (name, affiliation, email).

## Out-of-domain stress test

Protocol is in the PDF (`ood_protocol.tex`). To run scoring:

1. Add images under `ood/images/{domain}/`
2. Fill `ood/prompts.csv`
3. Run Locate-SAM2 and DINO on each pair (see `ood/README.md`)
4. Score with `ood/results_template.csv` → `results.csv`
5. Update empty cells in Table `tab:ood-results` in `ood_protocol.tex`

## v1 gaps

See `STATUS.md`. Remaining: **score OOD images** and fill results table; optional arXiv polish (acknowledgments, ORCID).
