"""Plots for matched synthetic fixtures; no claimed real-world superiority."""
def plot_comparison(results, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not results:
        raise ValueError('No policy results')
    signature = lambda item: (item['resource_unit'],item['resource_budget'],item['slot_ms'],item['horizon_ms'],
        [(row['job_id'],row['arrival_ms'],row['priority_class'],row['deadline_ms'],row['resource_units']) for row in item['jobs']])
    if any(signature(item) != signature(results[0]) for item in results[1:]):
        raise ValueError('Cannot compare different fixtures or resource budgets')
    labels = [item['policy'].replace('_','\n') for item in results]
    fig, axes = plt.subplots(1, 3, figsize=(12,4), layout='constrained')
    panels = [('completed','Completed jobs'),('deadline_met_fraction','Deadline met fraction'),('mean_wait_ms','Mean wait, completed jobs (ms)')]
    for ax, (key, title) in zip(axes, panels):
        values = [item['metrics'][key] for item in results]
        ax.bar(labels, [float('nan') if value is None else value for value in values], color=['#5276a7','#57917b','#bd8744','#935783'][:len(values)])
        ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
        for index, value in enumerate(values):
            if value is not None:
                ax.text(index, value, f'{value:.2f}', ha='center',va='bottom')
        ax.margins(y=.18)
    fig.suptitle(f"Synthetic fixture: n={results[0]['metrics']['jobs']} jobs; slot={results[0]['slot_ms']} ms; single deterministic replay (no uncertainty estimate)", fontsize=10)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
