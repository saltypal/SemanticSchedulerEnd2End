# Evaluation

Owns receiver queue/service lifecycle, episode execution, independent reference
labels, timing, coverage/ranking metrics, reconstruction metrics and persisted
traces. Scheduler policies select; the service mutates queues and completion
history. Parameter predictions and evaluator truth must remain separate.

Start at decoder_queue.py, decoder_service.py,
decoder_semantic_benchmark.py and codec_metrics/. Older simulation code was
retained from INFRA for reuse. Scheduler training is in Scheduler/training.
No new end-to-end validation was executed during this layout change.
