"""Compare only matched samples, versions, operating points and devices."""
import math
import statistics
from collections import defaultdict

def summarize_compatible(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = tuple(str(row[name]) for name in ('run_id','method','method_version','operating_point','device'))
        identity = (*key, str(row['image_id']))
        if identity in seen:
            raise ValueError(f'Duplicate measurement: {identity}')
        seen.add(identity)
        groups[key].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        item = dict(zip(('run_id','method','method_version','operating_point','device'),key))
        item['samples'] = len(group)
        item['failed'] = sum('status=failed' in str(row.get('notes','')) for row in group)
        item['image_ids'] = sorted(str(row['image_id']) for row in group)
        item['metrics'] = {}
        for metric in ('psnr_db','ssim','payload_bits','channel_uses','encode_time_ms','decode_time_ms','total_time_ms'):
            values = [float(row[metric]) for row in group if row.get(metric) not in (None,'') and math.isfinite(float(row[metric]))]
            item['metrics'][metric] = {'n':len(values), 'mean':statistics.fmean(values) if values else None,
                'std_across_images':statistics.stdev(values) if len(values)>1 else None,
                    'positive_infinite':sum(row.get(metric) not in (None,'') and float(row[metric]) == float('inf') for row in group)}
        summaries.append(item)
    return summaries

def plot_quality(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = summarize_compatible(rows)
    if not groups:
        raise ValueError('No benchmark rows')
    # Across-run matching is an explicit operation, never an implicit merge.
    if len({group['run_id'] for group in groups}) > 1 or len({group['device'] for group in groups}) > 1:
        raise ValueError('Select one run and device before plotting a comparison')
    samples = {tuple(group['image_ids']) for group in groups}
    if len(samples) > 1:
        raise ValueError('Comparisons require the same image IDs')
    labels = [f"{group['method']}\n{group['operating_point']}\nn={group['samples']}, failed={group['failed']}" for group in groups]
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis, metric, label in zip(axes, ('psnr_db','ssim'), ('PSNR (dB)','SSIM')):
        values = [group['metrics'][metric] for group in groups]
        axis.bar(labels,[value['mean'] if value['mean'] is not None else float('nan') for value in values],
                 yerr=[value['std_across_images'] or 0 for value in values],capsize=4)
        for index, value in enumerate(values):
            if value['positive_infinite']:
                axis.text(index, .5, f"PSNR = infinity\nLossless images: {value['positive_infinite']}",
                          ha='center', va='center', transform=axis.get_xaxis_transform())
            elif value['mean'] is None:
                axis.text(index, .5, 'Measurement unavailable', ha='center', va='center',
                          transform=axis.get_xaxis_transform())
        if all(value['mean'] is None for value in values):
            axis.set_yticks([])
        axis.set_ylabel(label)
        axis.grid(axis='y',alpha=.2)
    fig.suptitle('Kodak reconstruction; error bars = sample SD across images, not confidence intervals',fontsize=10)
    fig.savefig(destination,dpi=160)
    plt.close(fig)
