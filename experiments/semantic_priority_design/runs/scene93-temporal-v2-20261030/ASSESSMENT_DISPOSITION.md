# Revision assessment disposition

Assessment-only Sol Medium identified two concrete contradictions. Main corrected both:

1. Same-context probes now use PAIR_VIEW for both P training and pair evaluation: two displayed candidates, two displayed processed records, and only the displayed UE windows. Full-queue deployment is explicitly tested as generalization. D/E training remains full-context.
2. No extra no-history teacher call is claimed. Student temporal/history ablations reuse the existing accepted pair labels. The 216-call allocation is unchanged.

The assessor confirmed the scene dimensions, causal-window mechanism, preserved sixteen methods and D/E math during its initial static inspection. A focused second assessment was attempted but returned model-at-capacity; do not report a successful re-review. Main static checks verified the changes. No training, detector/VLM inference, data download or remote execution occurred.
