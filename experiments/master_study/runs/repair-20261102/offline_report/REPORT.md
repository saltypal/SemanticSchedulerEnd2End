# Saved-run report

development only; fixed first seed, no best-seed selection

Informative VLM guidance: inactive_all_neutral

| Method | D waiting penalty (lower better) |
|---|---:|
| spi_fifo | 3.374083 |
| spi_novelty | 3.434858 |
| ranknet | 3.630309 |
| deepsets_ranknet | 3.647492 |
| delay_huber | 3.671407 |
| lambdamart | 3.703640 |
| mlp_choice | 3.705926 |
| set_transformer_ranknet | 3.726756 |
| ranknet_warmstart_ppo | 3.746413 |
| temporal_deepsets_ranknet | 3.770175 |
| priority_classifier | 3.907849 |
| linear_choice | 4.159368 |
| listmle | 4.179957 |
| td_per | 4.713065 |
| relevance | 4.920339 |
| masked_double_dqn | 4.962876 |
| predicted_edf | 5.026631 |
| masked_ppo | 5.222276 |
| round_robin | 5.805900 |
| fifo | 5.822023 |
| random | 5.919610 |
| novelty | 6.191743 |

No teacher-feedback score is compared against D because historical workloads differ.
See inventory.csv for blocked/interrupted arms. These outputs do not establish final generalization, convergence or semantic correctness.
