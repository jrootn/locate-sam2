# What is still missing for a submission-ready paper

## Must complete before arXiv

| Item | Status | Action |
|------|--------|--------|
| Author affiliation + email | Placeholder | Edit `authors.tex` |
| OOD stress test scores | Empty table | Run `ood/` workflow, fill Table 6 |
| Proofread prose | Draft | Read `build/main.pdf` end-to-end |
| Acknowledgments | Missing | Add section if funding/sources apply |
| Figure page order | Fixed in v2 layout | Rebuild and verify no floats after refs |

## Strongly recommended

| Item | Status | Action |
|------|--------|--------|
| RefCOCO+ oracle baseline | Not run | Optional GT-box eval on RefCOCO+ |
| RefCOCO-g val | Not run | `run_eval.py --dataset refcocog` |
| Comparison to supervised RES | Mentioned only | Add citation table (DeRIS, etc.) without claiming win |
| SAM 3 positioning paragraph | Brief | Expand if targeting vision venue |
| Limitations figure / error taxonomy | Text only | Optional pie chart of failure tags |
| Supplementary PDF | None | Extra qual grid, per-sample logs |

## Optional polish

| Item | Notes |
|------|--------|
| ORCID + equal contribution footnote | For formal venues |
| `\usepackage{balance}` last page columns | Even column length |
| Graphic abstract | For social / website |
| Code release DOI + model cards | Reproducibility checklist |
| Human checklist inter-rater agreement | When OOD scored |

## Already in the PDF (v2 structure)

1. Introduction (contributions + scope)
2. Related Work (paragraphs)
3. Method (architecture fig + adapter + insight)
4. Experiments
   - 4.1 Setup
   - 4.2 Main results (tables + panel fig)
   - 4.3 Ablations
   - 4.4 Qualitative
   - 4.5 Failure analysis
   - 4.6 OOD stress test
5. Discussion (+ future work + limitations)
6. Conclusion
7. Reproducibility and licenses
8. References (after `\clearpage`)
