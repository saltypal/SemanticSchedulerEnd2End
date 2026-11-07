# Windows SPI comparison recovery

User authorized full comparison on saved 4000-image cohort, with 3933 usable VLM labels. No relabeling. Reuse exact v9 group/split assignments and enforce them before fitting. Fresh extraction required because published v9 banks are inaccessible; preserve v9 partial models/results unchanged.

Compare CLIP, SigLIP2, DINO, ConvNeXtV2 with existing vector/neural heads, four balancing strategies and supported grouped searches. Selection uses development metrics; after choices freeze, final test is evaluated. Existing per-balance locked-test receipts describe that stage, not completion of the final test.

Fix: center-based token masks remain unchanged unless empty; then use patch-rectangle intersection with valid image content. Same fallback applies to deployment profiling. This preserves all previous successful masks while supporting extreme aspect ratios. Medium assessment passed AST/embedded-source checks and21 representative CPU shape cases; no GPU inference claimed from these checks.

Notebook SPI_4000_Comparison.ipynb uses private Windows managed labels input; same notebook identity satyapaladugu/crisismmd1000-spi-labeling new version. Results under experiments/spi_comparison_4000/runs/UTC. Final portable archive contains model artifacts and metrics, excludes media/checkpoint shards and heavy feature banks. Preserve partial profiles as partial and any failures explicitly. Targets remain mixed-protocol VLM pseudo-labels.

Evaluation scope: these native test groups were already evaluated in the provisional Mini classifier study. They remain excluded from fitting and combination selection, but are development-exposed and are not a new independent final test. Report all comparisons with that limitation; retain a fresh final cohort for confirmatory claims.

## Frozen-split repair — 2026-11-01
Version10 failed before fitting because preparing data regenerated development assignments. prepare now accepts frozen_records and restores exact saved group/split values after validating IDs, labels and image hashes; it rejects group, exact-hash or recomputed parent-component leakage across roles. The notebook requires saved records and returns from preparation before any split generator. Local fixture validated exact assignment preservation and rejection of cross-role group leakage. Labels and Mini dataset remain unchanged. Version11 submission pending operational receipt.
