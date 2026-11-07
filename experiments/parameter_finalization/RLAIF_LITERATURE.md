# AI feedback and its role in SPI —28October2026

Current pipeline is VLM pseudo-labeling plus supervised classifier distillation,
not reinforcement learning. AI critique alone does not make a system RLAIF.
Preference-learning papers sometimes use RLAIF broadly, including DPO rather
than an online RL loop. Report the actual training objective precisely.

## Primary papers

1. Lee etal2023, RLAIFvsRLHF: https://arxiv.org/abs/2309.00267
   AI preferences train reward models used for RL; direct-RLAIF also obtains
   rewards directly from the labeler. Relevant if a future scheduler uses AI
   feedback as an RL reward. Not the current classifier training objective.
2. Yu etal2024/CVPR2025, RLAIF-V: https://arxiv.org/html/2405.17220v3
   Split candidate responses into atomic claims, score image-grounded yes/no
   questions, construct preferred/rejected pairs, and iteratively train viaDPO.
   Best immediate inspiration is claim-level diagnosis, not another global
   priority guess. Claimed gains are on their benchmarks, not our SPI task.
3. Li etal2023/EMNLP2024, Silkie: https://arxiv.org/abs/2312.10665
   AI feedback on outputs from12LVLMs, assessing visual faithfulness and other
   criteria; distill preferences withDPO. Supports keeping faithfulness distinct
   from willingness to assign a priority. Its GPT4V labeling cost is not our
   free local workflow.
4. Xiong etal2024/CVPR2025, LLaVA-Critic: https://arxiv.org/abs/2410.02712
   A trained multimodal evaluator supplies scores/pairwise comparisons. A useful
   judge candidate, not independent ground truth or proven SPI expertise.
5. Rafailov etal2023, DPO: https://arxiv.org/abs/2305.18290
   Preference optimization without a separately trained reward model or PPO.
   Not a drop-in replacement for ordinary four-class cross-entropy training.

## Bounded recommendation

Keep current300model comparison unchanged. Preserve evidence clauses, labels,
abstentions and timing. Diagnose factual propositions separately from urgency:
Does water cover the bridge deck? Is a trapped person visibly present? Review
support/contradiction/indeterminate with native labels where actually compatible.
Do not turn agreement among models into truth. Avoid prompting judges with a
preferred class. Maintain independent split/control boundaries and negative
cases. A more talkative model or fewer unknowns is not necessarily more accurate.

A claim verifier is a conditional follow-up, not implemented or launched by
this note. Do not add VLMfine-tuning, PPO, DPO or scheduler redesign during the
current parameter milestone. For the tiny SPIhead, use supervised loss on
accepted labels. Scheduler preference/ranking work remains separate ownership.
