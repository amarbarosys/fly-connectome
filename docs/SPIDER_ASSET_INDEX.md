# Spider Asset Index

Generated 2026-10-03  |  ~900 MB on disk under `.work/spider-docs/` (gitignored, re-fetch steps below)

No spider connectome exists, so the spider circuit is a hypothesis model built on behaviour,
anatomy, cell types and web physics. Two datasets already carry passing checks
(`tools/spider_turn_test.py` A-D): Hung 2026 and Mortimer 2019.

## Behaviour and physics data

| Size | Asset | Source | Used by / for |
|-----:|:------|:-------|:--------------|
| 2.0 MB | `hung-repo/5_localization/processed/` | Hung 2026 PLoS CB, GordusLab/spider_prey_capture_paper | checks A, B: 10 turns + 26 radii x 7150 frames vibration, Uloborus |
| 3.0 MB | `hung-repo/1_webvibration/data/FFT_table_*.csv` | same | web spectra prey/control, vibration-world calibration |
| 73 MB | `hung-repo/2_wavelet_hmm/data/*.npz` | same | leg-joint wavelet spectra, behaviour-state reference |
| 64 KB | `hung-repo/4_glmhmm/data/F1s_*.npy` | same | GLM-HMM scores, feedback-loop reference |
| 32 MB | `hung-videos/s006-s010.mp4` | Hung 2026 supplementary S1-S5 videos (PLoS) | S4 (`s009`) decoded with plot-scrape to `s4_labels.csv` and `s4_timeline.csv` (adds the pure-fly vibration z-score): 2 episodes, manual + GLM state per 0.1 s, colour->state inferred from Fig 5C legend (cyan static, straw crouch, orange high-freq), manual vs GLM 80-82%. S5 (`s010`) extracted: `s5_radii_traces.csv` (5 radii, ~7.5 s), `s5_spider_heading.csv`, notes `s5_extraction.md` |
| 9.8 MB | `mortimer2019/Data/` (90 txt) | Mortimer 2019 J R Soc Interface, Dryad 10.5061/dryad.7rm572m | checks C, D: FEM web, 8 hub legs, 14 radii x mass, 4 distances x 3 strengths, Araneus + Zygiella |
| 77 MB | `dahl2025/STUDY.mat` + 3 `.m` | Dahl 2025 eLife, OSF gpnct | individual recognition, P. regius; readme names it `SUBJ.mat` |
| 565 MB | `rem2022/Retina3.0-DLC.zip` | Rossler 2022 PNAS, Zenodo 6616655 | DeepLabCut project: config, 24 model files, 841 labelled frames; tracks retinal tubes + body |
| 27 KB | `rem2022/REMSleep.rmd`, `REM_retinaAnalysis.m` | same | analysis code; needs datasets S2/S3 |
| 413 MB | `rem2022/supp/` | same, Europe PMC supplementaryFiles | 9 movies (sm01-09.mp4) + SI appendix PDF + fig 1; NO datasets S2/S3 (retinal-movement CSVs not in this bundle, PNAS hosts them separately); movies can feed the DLC project |

## Papers (full text; `PMC*.txt` plain body, JATS, or BioC XML)

| PMCID | Paper | Topic |
|:------|:------|:------|
| PMC13592698 | Hung 2026 PLoS CB | Uloborus vibration prey capture, test set |
| PMC12919972 | Artiushin 2026 eLife | Uloborus 3D brain atlas (code: GordusLab/Artiushin-elastix-eLife) |
| PMC10697844 | Jin 2023 Nat Ecol Evol | spider brain scRNA atlas, cell types (GEO GSE241696, not downloaded) |
| PMC11418026 | De Agro 2024 JEB | jumping spider multi-eye motion (zenodo 8393472, not downloaded) |
| PMC8046691 | Barth 2021 JCP-A | sensory guidance, slit sensilla |
| PMC9388130 | Rossler 2022 PNAS | REM-like sleep, retinal movements |
| PMC5940701 | Nabawy 2018 Sci Rep | jump trajectories, P. regius |
| PMC12604854 | Dahl 2025 eLife | individual recognition |
| PMC7234261 | Aguilar-Arguello 2020 Curr Zool | detour distance |
| PMC3181266 | Girard 2011 PLoS One | peacock spider vibrational courtship |
| book ch. | Winsor, Morehouse & Jakob 2023, Distributed Vision ch. 10 (doi 10.1007/978-3-031-23216-9_10) | spider vision review: eye roles, retina numbers, secondary->MB and principal->arcuate pathways; local PDF `~/Downloads/Distributed_Vision_Spiders.pdf`, text `winsor2023.txt`, notes `winsor2023-vision-spec.md` |

The first checked spider evidence packet is built by `tools/spider_corpus.py` into
`.work/papers/spider/`. It imports these local texts without modifying the
structural-plasticity corpus, records input and cleaned-text SHA-256 digests,
deduplicates DOI/PMCID, keeps section-based passage locators, and excludes JATS
reference lists. Portia detour planning (Cross & Jackson 2016, DOI
10.1002/jeab.189, PMCID PMC4751061) is now locally available as an NCBI BioC
full-text export. The index's prior “no text” note below is superseded for this
paper. The checked count is 10 primary sources plus 2 reviews; Steck 2024
(PMC11756594) remains unavailable and is not counted. See
[`docs/plans/SPIDER_EVIDENCE.md`](plans/SPIDER_EVIDENCE.md) for packet scope and
the reproducible gate.

Not open / not found: Homann 1928 (eye-patch roles; behaviour now covered by Land 1971 / Zurek & Nelson 2012 via Winsor 2023),
Steck 2024 (PMC11756594, no text), 2013 reversal learning, 2021 silk toughness.

## Notes and search logs

| Asset | What |
|:------|:-----|
| `INDEX.md` | running source log with pass thresholds pulled from Hung 2026 |
| `video-kRQMOF5c2Z8.md` | claims from the jumping-spider video, paraphrased, to verify against papers |
| `video-assets-search.txt`, `../online-search.txt`, `../europepmc.txt` | raw search hits |
| `hung-tree.tsv` | Hung repo file list (sizes), for the sparse clone |
| `../orbweaver-spec.md`, `../spider-lit.md` | circuit spec v0, literature sweep |

## Re-fetch

- Hung: `git clone --depth 1 --filter=blob:none --sparse https://github.com/GordusLab/spider_prey_capture_paper .work/spider-docs/hung-repo` then `git sparse-checkout set --no-cone '/*' '!/deeplabcut_beh_analysis/dlc-models/' '!*.keras' '!*tfevents*'` (107 MB of 1.6 GB).
- Mortimer: browser download from the Dryad page (API needs a token), unzip into `mortimer2019/`.
- Dahl: `https://api.osf.io/v2/nodes/gpnct/files/osfstorage/` download links.
- Rossler: `https://zenodo.org/api/records/6616655` file links.
- Papers: `https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML`.
