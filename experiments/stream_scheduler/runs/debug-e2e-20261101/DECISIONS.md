# Repair decisions and evidence boundary

- Preserve v4 results. No old malformed teacher output is guessed into a valid label.
- Enforce task-specific teacher JSON during token generation using lm-format-enforcer0.11.3; keep post-parse validation. Named aspect objects avoid an observed nullable-array grammar fault; normalize them to four ratings/unknown for downstream models. Official pinned Qwen tokenizer reachability and EOS checks pass for all four output types on CPU. Actual VLM verification is a separate bounded GPU run.
- Keep the original teacher acceptance thresholds. Expand the bounded request allocation to28 train/12 validation candidate contexts and28/12 prefix contexts, retain24 replay calls and32 controls:216calls. No locked-test teacher queries. More questions provide rejection headroom, not guaranteed valid supervision.
- Always persist a teacher-arm blocked/failed outcome; do not exit before the requested model ledger exists.
- Record child process failures and unfinished statuses in a separate run_outcome.json. Process completion is not convergence.
- Preserve source identity when churn creates the initial backlog. Otherwise local-repeat experiments are invalid.
- Freeze the initial Deep Sets warmstart before resuming that producer. Bind RL stream identities to the environment source hash; changed simulator code is not an exact resume of old v4 training.
- Batched independent queue updates preserve pooling and loss definitions; TD-PER uses pure normalized TD priorities, VLM replay keeps its uniform mixture. Exploration scales to half the requested action budget and is persisted across resume.
- Reject nonfinite simulation inputs and malformed model shapes/preprocessing at load time. Report empty-episode normalized metrics as undefined.
- Bound native BLAS threads and record actual thread pools in profiling. Reuse hash-verified cached mission/image embeddings instead of rerunning vision extraction for scheduler checks.

## Remaining scientific requirements

The current reference tiers are VLM silver labels. Narrow person/tie/blank controls do not certify emergency priority. Real video chronology, natural cross-view correspondence, independent final urgency assessment and full-budget multiseed convergence remain unestablished. No actual codec inference is introduced. CPU fixtures exercise all model mechanisms including feedback; they do not replace qualified real VLM labels.
