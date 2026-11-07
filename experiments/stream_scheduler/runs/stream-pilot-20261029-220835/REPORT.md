# Local synthetic stream pilot

Synthetic parameters/world labels only. No SPI classifier, RGB inference, Qwen or codec was run.
These short training updates test execution; they do not establish convergence or emergency importance.

| Policy | Episode | Max UE pool | J_time | Critical capped delay |
|---|---:|---:|---:|---:|
| fifo | 0 | 4 | 0.3711 | 6.697938554172344 |
| fifo | 1 | 60 | 0.0897 | None |
| fifo | 2 | 100 | 0.0908 | 9.699162162173318 |
| rr | 0 | 4 | 0.3769 | 5.54712201421614 |
| rr | 1 | 60 | 0.0963 | None |
| rr | 2 | 100 | 0.1388 | 8.472545301010497 |
| random | 0 | 4 | 0.4850 | 2.0970556803602554 |
| random | 1 | 60 | 0.1942 | None |
| random | 2 | 100 | 0.1190 | 8.748418777397786 |
| spi | 0 | 4 | 0.5896 | 1.5042252391781785 |
| spi | 1 | 60 | 0.1968 | None |
| spi | 2 | 100 | 0.0578 | 7.158291344102741 |
| relevance | 0 | 4 | 0.4458 | 5.54712201421614 |
| relevance | 1 | 60 | 0.1515 | None |
| relevance | 2 | 100 | 0.1454 | 10.0 |
| novelty | 0 | 4 | 0.4153 | 6.3360458275725104 |
| novelty | 1 | 60 | 0.1673 | None |
| novelty | 2 | 100 | 0.0933 | 10.0 |
| linear_choice | 0 | 4 | 0.4703 | 2.2637769753210994 |
| linear_choice | 1 | 60 | 0.1327 | None |
| linear_choice | 2 | 100 | 0.1061 | 8.746134911713659 |
| mlp_choice | 0 | 4 | 0.4982 | 1.4376358588005922 |
| mlp_choice | 1 | 60 | 0.2344 | None |
| mlp_choice | 2 | 100 | 0.1326 | 7.523218927297182 |
| ranknet | 0 | 4 | 0.4982 | 1.4376358588005922 |
| ranknet | 1 | 60 | 0.2344 | None |
| ranknet | 2 | 100 | 0.1338 | 7.523218927297182 |
| lambdamart | 0 | 4 | 0.4883 | 2.539373565228715 |
| lambdamart | 1 | 60 | 0.3195 | None |
| lambdamart | 2 | 100 | 0.1387 | 10.0 |
| priority_classifier | 0 | 4 | 0.4982 | 1.4376358588005922 |
| priority_classifier | 1 | 60 | 0.2344 | None |
| priority_classifier | 2 | 100 | 0.1295 | 7.389071517080085 |
| masked_ppo | 0 | 4 | 0.4993 | 1.4376358588005922 |
| masked_ppo | 1 | 60 | 0.2079 | None |
| masked_ppo | 2 | 100 | 0.1353 | 7.523218927297182 |
| masked_double_dqn | 0 | 4 | 0.4982 | 1.4376358588005922 |
| masked_double_dqn | 1 | 60 | 0.2135 | None |
| masked_double_dqn | 2 | 100 | 0.1429 | 8.777756331599281 |
| td_per | 0 | 4 | 0.4982 | 1.4376358588005922 |
| td_per | 1 | 60 | 0.2135 | None |
| td_per | 2 | 100 | 0.1429 | 8.777756331599281 |

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
