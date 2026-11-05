"""Explicit partial neural-operation profiling, separate from timed extraction."""
import torch


def profile_forward(module,args=(),kwargs=None):
    """Counts only operators supported by torch.profiler, never total model FLOPs."""
    activities=[torch.profiler.ProfilerActivity.CPU]
    if any(p.is_cuda for p in module.parameters()):activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.inference_mode(),torch.profiler.profile(activities=activities,with_flops=True,record_shapes=True) as trace:
            module(*args,**(kwargs or {}))
        rows=[{'operator':e.key,'flops':e.flops or None,'calls':e.count} for e in trace.key_averages()]
        counted=sum(r['flops'] or 0 for r in rows)
        return {'status':'partial_supported_ops' if counted else 'unsupported',
            'supported_operator_flops':counted if counted else None,
            'uncounted_operators':[r['operator'] for r in rows if r['flops'] is None],
            'operators':rows,'convention':'PyTorch profiler formula, matmul multiply/add counted separately',
            'total_model_flops':None,'latency_scope':'instrumented diagnostic, not inference latency'}
    except Exception as error:
        return {'status':'failed','supported_operator_flops':None,'total_model_flops':None,
                'reason':type(error).__name__+': '+str(error)}
