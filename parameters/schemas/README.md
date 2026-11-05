# Draft parameter-validation JSON contracts

These schemas accompany docs/JSON_PARAMETER_VALIDATION.md. They specify the
planned audit interchange,not the existing ParameterBundle or BSE1 wire layout.
No runtime migration/model experiment is performed by adding them.

- parameter_audit.schema.json: no reference labels/reviews in runtime values;
  unavailable implies null+reason; available zero is valid.
- qwen_claim_review.schema.json: explicit execution failures/not-run state;
  completed reviews require identified image inputs,model,prompt,raw response,
  declared blinding,and model_assessed claim verdicts.
  Completed reviews also require separate binary consistency/sufficiency results
  and a frozen sufficiency rubric ID. A numeric score alone has no implied visual
  fact; reviewer decisions are limited to answerable claims and required facts.
  Primary visible_parameter_json reviews show candidate values; optional blinded
  atomic reviews hide them. Both hide external CLIP diagnostics and reference labels.

Actual finite values,unique parameter names,model/image hashes,join completeness,
blinding receipts,and causal history invariants also require deterministic code.
A schema can check declarations,not prove the reviewer actually received images
or that labels were withheld. Use strict JSON serialization (allow_nan=False).

../examples/parameter_audit.jsonl and ../examples/qwen_not_run.json are illustrative contract
fixtures. They are excluded from measured summaries. No model verdicts are invented.

Future records must inventory all applicable candidates,with unavailable entries
for unsupported evidence. The minimal fixture shows only novelty and a missing
prominence field. The adapter from current names such as relevance/rarity/spatial
to the audit names must be versioned; do not silently change existing APIs.
