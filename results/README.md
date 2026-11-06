# Results

`chrono/` and `season/` hold the outputs of the Colab run of November 2026 (`.csv`, `.png`, `.txt`, the trained checkpoints `.pt` and the resume caches `.pkl`). To regenerate them, run the notebooks and copy the run folders here:

| Folder | Source |
|---|---|
| `chrono/` | notebook 01, `MyDrive/PINN-LEWS-HYDROL86065_results/chrono/` |
| `season/` | notebook 02, `MyDrive/PINN-LEWS-HYDROL86065_results/season/` |

- **Copy:** all `.csv`, `.png` and `.txt` files.
- **Keep:** the `.pt` checkpoints (main run and LSTM) and the `.pkl` resume caches. They are small (about 20 MB in total) and let reviewers check the results without re-training; notebooks 01 and 02 load them by default (`REUSE_STORED = True`).

`architecture_search/` holds the tables of the network search: `arch_stage1.csv` (random screen, 1 seed), `arch_stage2.csv` (shortlist × 3 seeds) and `arch_selection_table.csv` (mean ± SD and the selection rule). The search was run on the Nov 2025 – Jan 2026 part of the Dev108 record with the corrected pipeline.
