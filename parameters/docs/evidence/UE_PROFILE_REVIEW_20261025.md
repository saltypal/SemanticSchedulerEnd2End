# UE task-profile drafting and review

User requested Luna task-profile drafting for the parameter/VLM pilot. Luna High
returned six operator-intent drafts without editing code or inspecting/labeling
images. Main reviewed and integrated the task panel into config/ue_task_profiles.json
and docs/VLM_PILOT_PROPOSAL.md.

Accepted: six simulatedUEs,distinct operator purposes,two building-inspection
senders with independent histories,road-condition contrasts,phone tasks,no fixed
importance weights,and unknown-reference handling.

Corrections before integration:

- Make both buildingUEs reference one identical MissionSpec/text/version. Different
  paragraphs would confound a test intended to isolate history effects.
- Do not map generic LADI roads_damage to blocked road or to a finer surface-only
  damage target without exact native support. Keep blockage and damage separate.
- Replace generic "ordinary everyday documentation" with a measurable household
  inventory task. Object presence still does not certify absence of incidents.
- Replace an ambiguous optional-informativeness/humanitarian OR rule with a clearly
  declared incident-informativeness task. MEDIC supports its native label,not an
  automatically expanded urgency/distress claim.
- Keep short positive task texts separate from hard-negative control descriptions.
  Negative prompts are not incorporated as positive task embeddings.

This is operator-defined test data,not a learned profile or validated real-user
intent study. No vision-inference accuracy follows from profile drafting.
