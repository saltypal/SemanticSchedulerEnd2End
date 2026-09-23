"""Legacy scatter plot with compatibility guards and explicit missing data."""
import math
from pathlib import Path

def plot_quality_vs_channel_uses(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows:
        raise ValueError('No measurements to plot')
    for key in ('run_id','device','snr_definition'):
        values = {row.get(key) for row in rows if row.get(key) is not None}
        if len(values)>1:
            raise ValueError(f'Mixed {key}; select a compatible run first')
    fig,axis=plt.subplots(figsize=(8,5))
    missing=0
    for method in sorted({str(row['method']) for row in rows}):
        valid=[]
        for row in rows:
            if str(row['method']) != method: continue
            quality=row.get('psnr_db',row.get('psnr'))
            uses=row.get('channel_uses')
            failed=row.get('success') is False or 'status=failed' in str(row.get('notes',''))
            if failed or uses is None or quality is None or not math.isfinite(float(quality)):
                missing+=1
                continue
            valid.append((float(uses),float(quality)))
        if valid:
            axis.scatter([value[0] for value in valid],[value[1] for value in valid],label=f'{method} (n={len(valid)})')
    axis.set_xlabel('Complex channel uses per image')
    axis.set_ylabel('PSNR (dB)')
    axis.set_title(f'Individual observations; excluded failed/missing/nonfinite: {missing}')
    if axis.collections: axis.legend()
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(target,dpi=160)
    plt.close(fig)
    return target
