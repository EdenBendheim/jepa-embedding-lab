"""Render aggregate training dynamics from the verified October 14 journals."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from jepa_lab.training_diagnostics import summarize_training


def prepare(directory,results_path):
    results=json.loads(Path(results_path).read_text());series=[]
    for record in results['records']:
        name=f'mask{record["mask_size"]}-seed{record["training_seed"]}'
        path=Path(directory)/'training'/f'{name}.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['training_report_sha256']:
            raise ValueError('Local journal bytes differ from the published results')
        report=json.loads(path.read_text())
        if {k:v for k,v in report.items() if k not in ('history','limitation')}!=record['training']:
            raise ValueError('Journal recipe/identity differs from the published run')
        series.append(dict(name=name,mask_size=record['mask_size'],training_seed=record['training_seed'],
            journal_sha256=record['training_report_sha256'],**summarize_training(report,window=100)))
    return dict(version=1,date='2026-10-14',source_results_sha256=hashlib.sha256(Path(results_path).read_bytes()).hexdigest(),
        series=series,limitation='Non-overlapping 100-step window means; ribbons are the range across three seeds, not confidence intervals. Token spread/loss are not representation-quality evidence.')


def render(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    metrics=('loss_mean','target_token_std_mean')
    for size,color in ((2,'#2563eb'),(4,'#ea580c')):
        records=[r for r in summary['series'] if r['mask_size']==size]
        if len(records)!=3:raise ValueError('The published experiment requires three seeds per mask')
        x=[r['last_step'] for r in records[0]['windows']]
        if any([r['last_step'] for r in item['windows']]!=x for item in records):raise ValueError('Training windows do not align')
        for ax,metric in zip(axes,metrics):
            ys=[[row[metric] for row in item['windows']] for item in records]
            average=[mean(values) for values in zip(*ys)]
            ax.plot(x,average,label=f'{size}×{size} target',color=color,linewidth=2)
            ax.fill_between(x,[min(v) for v in zip(*ys)],[max(v) for v in zip(*ys)],color=color,alpha=.15)
    for ax,title,ylabel in zip(axes,('Latent prediction loss','EMA target token spread'),('Smooth L1 loss','Mean token standard deviation')):
        ax.set(title=title,xlabel='Completed optimizer steps',ylabel=ylabel)
        ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('Longer JEPA training: 100-step means and ranges across three seeds')
    fig.savefig(output);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('runs/budget-20261014'))
    p.add_argument('--results',type=Path,default=Path('experiments/budget-2026-10-14-results.json'))
    p.add_argument('--output',type=Path,default=Path('experiments/budget-2026-10-14-dynamics.svg'))
    p.add_argument('--summary',type=Path,default=Path('experiments/budget-2026-10-14-dynamics.json'))
    args=p.parse_args()
    try:
        summary=prepare(args.directory,args.results)
        render(summary,args.output)
        args.summary.write_text(json.dumps(summary,indent=2)+'\n')
    except (ValueError,OSError,KeyError,TypeError) as exc:p.error(str(exc))

if __name__=='__main__':main()
