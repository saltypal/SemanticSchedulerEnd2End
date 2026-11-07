# Local synthetic stream pilot

Synthetic parameters/world labels only. No SPI classifier, RGB inference, Qwen or codec was run.
These short training updates test execution; they do not establish convergence or emergency importance.

| Policy | Episode | Max UE pool | J_time | Critical capped delay |
|---|---:|---:|---:|---:|
| fifo | 0 | 4 | 0.3638 | 6.697938554172344 |
| fifo | 1 | 60 | 0.0857 | None |
| fifo | 2 | 100 | 0.0836 | 9.699162162173318 |
| rr | 0 | 4 | 0.3667 | 5.54712201421614 |
| rr | 1 | 60 | 0.0945 | None |
| rr | 2 | 100 | 0.1313 | 8.472545301010497 |
| random | 0 | 4 | 0.4652 | 2.0970556803602554 |
| random | 1 | 60 | 0.1887 | None |
| random | 2 | 100 | 0.1057 | 8.748418777397786 |
| spi | 0 | 4 | 0.4818 | 1.8267249634669445 |
| spi | 1 | 60 | 0.2455 | None |
| spi | 2 | 100 | 0.1136 | 5.252386536216591 |
| relevance | 0 | 4 | 0.3694 | 5.54712201421614 |
| relevance | 1 | 60 | 0.1766 | None |
| relevance | 2 | 100 | 0.1587 | 9.910563003219124 |
| novelty | 0 | 4 | 0.4010 | 6.3360458275725104 |
| novelty | 1 | 60 | 0.1621 | None |
| novelty | 2 | 100 | 0.0862 | 10.0 |
| linear_choice | 0 | 4 | 0.2751 | 7.3184248668446275 |
| linear_choice | 1 | 60 | 0.2547 | None |
| linear_choice | 2 | 100 | 0.1751 | 9.394457832490264 |
| mlp_choice | 0 | 4 | 0.3365 | 6.603712642740467 |
| mlp_choice | 1 | 60 | 0.2200 | None |
| mlp_choice | 2 | 100 | 0.1773 | 9.084767114786635 |
| ranknet | 0 | 4 | 0.3365 | 6.603712642740467 |
| ranknet | 1 | 60 | 0.2338 | None |
| ranknet | 2 | 100 | 0.1735 | 9.244660719321786 |
| lambdamart | 0 | 4 | 0.5112 | 2.013859276444389 |
| lambdamart | 1 | 60 | 0.2799 | None |
| lambdamart | 2 | 100 | 0.1750 | 10.0 |
| priority_classifier | 0 | 4 | 0.3050 | 7.199760322509924 |
| priority_classifier | 1 | 60 | 0.2123 | None |
| priority_classifier | 2 | 100 | 0.1735 | 9.244660719321786 |
| masked_ppo | 0 | 4 | 0.2225 | 8.016372250340742 |
| masked_ppo | 1 | 60 | 0.2554 | None |
| masked_ppo | 2 | 100 | 0.1729 | 9.715881863189537 |
| masked_double_dqn | 0 | 4 | 0.3050 | 7.199760322509924 |
| masked_double_dqn | 1 | 60 | 0.2123 | None |
| masked_double_dqn | 2 | 100 | 0.1601 | 9.340593127024842 |
| td_per | 0 | 4 | 0.3463 | 6.603712642740467 |
| td_per | 1 | 60 | 0.2123 | None |
| td_per | 2 | 100 | 0.1601 | 9.340593127024842 |

## Learners

- linear_choice: fixture_trained
- mlp_choice: fixture_trained
- ranknet: fixture_trained
- lambdamart: fixture_trained
- priority_classifier: fixture_trained
- masked_ppo: fixture_trained
- masked_double_dqn: fixture_trained
- td_per: fixture_trained
- rl_vlm_f: blocked
- vlm_replay: blocked
- visionprefer: blocked

## Limits

The profile/trajectory stream is a constructed fixture, not coherent real camera imagery.
Timing uses synthetic service units; measured selection milliseconds are host CPU measurements only.
Header bytes are compact JSON interchange bytes, not the current 35-byte wire encoding.
No claimed FLOPs for teacher/extractor; dense matrix MAC estimates exclude preprocessing and activations.
Real VLM-dependent arms are explicitly blocked until qualified feedback exists.
