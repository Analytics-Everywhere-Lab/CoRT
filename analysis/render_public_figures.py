#!/usr/bin/env python3
"""Optional data-only plots of Figures 3-8; no manuscript media are read."""
import csv
import os
from pathlib import Path
import signal
import tempfile

from reproduce_all import verified_receipt

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
COLORS = {'F':'#3f6fe5','K':'#e77833','H':'#3f6fe5','V':'#238b73'}


def rows(n):
    with (ROOT/'figures'/f'figure{n}_data.csv').open() as stream:
        return list(csv.DictReader(stream))


def errorbar(ax, row, x, color, label=None):
    estimate, low, high = map(float, (row['estimate'], row['ci_low'], row['ci_high']))
    ax.errorbar(x, estimate, yerr=[[estimate-low],[high-estimate]], marker='o',
                color=color, capsize=3, label=label, linestyle='none')


def finish(plt, figure, number, note):
    for ax in figure.axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#e5e5e5',linewidth=.6)
        ax.set_axisbelow(True)
    figure.suptitle(f'Figure {number}: Public data reconstruction', fontsize=13)
    figure.text(.5,.01,note,ha='center',fontsize=9)
    figure.tight_layout(rect=(0,.06,1,.94))
    dest = ROOT/'figures/rendered'
    dest.mkdir(exist_ok=True)
    figure.savefig(dest/f'figure{number}.png',dpi=160,metadata={'Software':'CoRT public aggregate renderer'})
    plt.close(figure)


def main():
    signal.alarm(120)
    temporal_note = ('Replayed descriptive CIs from anonymous participant vectors.'
                     if verified_receipt() else
                     'Saved descriptive CIs; run reproduce_all.py to verify replay.')
    with tempfile.TemporaryDirectory(prefix='cort-mpl-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
        fig, axes = plt.subplots(1,2,figsize=(10,4.5))
        data = rows(3)
        for ax, model in zip(axes,('gemma','qwen')):
            for op, label in [('single_full_frame','F'),('complement_once','K')]:
                selected = [r for r in data if r['model']==model and r['operator']==op]
                layers = sorted({int(r['layer']) for r in selected})
                for corrupt in ('blank_full','wrong_time'):
                    for direction in ('rescue','damage'):
                        curve = sorted([r for r in selected if r['corruption']==corrupt and r['direction']==direction],key=lambda r:int(r['layer']))
                        ax.plot([int(r['layer']) for r in curve],[float(r['rho']) for r in curve],color=COLORS[label],alpha=.25,linewidth=1)
                means = [sum(float(r['rho']) for r in selected if int(r['layer'])==layer)/4 for layer in layers]
                ax.plot(layers,means,'o-',color=COLORS[label],label=label,linewidth=2)
            ax.axhline(1,color='#888888',linestyle='--',linewidth=1)
            ax.set(title=model.capitalize(),xlabel='Post-block cut',ylabel='Recovery')
            ax.legend(frameon=False)
        finish(plt,fig,3,'80 exploratory clusters. Thick curves average four cells; no intervals shown.')

        fig, axes = plt.subplots(1,2,figsize=(10,4.5))
        data = rows(4)
        for ax, model in zip(axes,('gemma','qwen')):
            for i, op in enumerate(('early_F','early_K','late_F','late_K')):
                row = next(r for r in data if r['model']==model and r['operator']==op)
                errorbar(ax,row,i,COLORS[op[-1]])
            ax.set_xticks(range(4),('Early F','Early K','Late F','Late K'))
            title = 'Gemma: fixed-cut confirmation' if model == 'gemma' else 'Qwen: descriptive comparison'
            ax.set(title=title,ylabel='Recovery')
        finish(plt,fig,4,'46 clusters. Historical static marginal CIs; paired contrast bounds are separate.')

        fig, axes = plt.subplots(1,3,figsize=(12,4.5),sharey=True)
        data = rows(5)
        for ax,task,title in zip(axes,('current_state','new_interaction','interaction_order'),('Current object','New interaction','Interaction order')):
            for j, role in enumerate(('H','V','K')):
                for i,cut in enumerate((22,34)):
                    row=next(r for r in data if r['task']==task and r['direction']=='B0->B4' and int(r['cut'])==cut and r['position_set']==role)
                    errorbar(ax,row,i+(j-1)*.2,COLORS[role],role if i==0 else None)
            ax.set_xticks([0,1],['L22','L34'])
            ax.set(title=title,ylabel='Recovery')
            ax.legend(frameon=False)
        finish(plt,fig,5,'Clip to repeated final frame. ' + temporal_note)

        fig, axes = plt.subplots(1,2,figsize=(11,6),sharex=True,sharey=True)
        data = rows(6)
        labels=['C','Q','P','R','CQ','CP','CR','QP','QR','PR','CQP','CQR','CPR','QPR','CQPR']
        for ax, model in zip(axes,('gemma','qwen')):
            for i,label in enumerate(labels):
                row=next(r for r in data if r['model']==model and r['subset']==label)
                e,lo,hi=map(float,(row['estimate'],row['ci_low'],row['ci_high']))
                ax.errorbar(e,i,xerr=[[e-lo],[hi-e]],color='#277da1',marker='o',capsize=2)
            ax.set_yticks(range(15),labels)
            ax.set(title=model.capitalize(),xlabel='Recovery')
        axes[0].invert_yaxis()
        finish(plt,fig,6,'All 15 nonempty static role subsets; historical replayed descriptive CIs.')

        fig, axes = plt.subplots(1,2,figsize=(10,4.5),sharey=True)
        data = rows(7)
        for ax,task,title in zip(axes,('new_interaction','interaction_order'),('New interaction','Interaction order')):
            for i,role in enumerate(('C+Q+P+R','Q+P+R','C+Q+R')):
                row=next(r for r in data if r['task']==task and r['position_set']==role)
                errorbar(ax,row,i,('#277da1','#d1495b','#599c52')[i])
            ax.set_xticks(range(3),['All K','Without C','Without P'])
            ax.set(title=title,ylabel='Recovery',ylim=(0,.8))
        finish(plt,fig,7,'Gemma L34, clip to repeated. ' + temporal_note)

        fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True)
        data=rows(8)
        families={
            'combined':[('Fhalf_Chalf_L34_prune__readout_fused_seed','Half F/C + readout'),
                        ('F0_L30_prune__CQ_fused_seed','Remove F at L30 + CQ'),
                        ('F0_L30_prune__readout_fused_seed','Remove F at L30 + readout')],
            'followup':[('native_readout_seed','No removal + readout'),
                        ('F133_L34_readout_seed','Half F + readout')]}
        for ax,session,title in zip(axes,('combined','followup'),('Removal-trained session','Scene-only session')):
            native=next(r for r in data if r['session']==session and r['condition']=='native')
            ax.scatter(100,float(native['accuracy_percent']),marker='*',color='#333333',s=85,label='Native')
            for j,(prefix,label) in enumerate(families[session]):
                color=('#277da1','#d1495b','#599c52')[j]
                for seed in (17,29):
                    row=next(r for r in data if r['session']==session and r['condition']==prefix+str(seed))
                    assert row['timing_status']=='measured'
                    ax.scatter(float(row['request_percent_native']),float(row['accuracy_percent']),
                               edgecolors=color,facecolors=color if seed==17 else 'white',
                               marker=('o','s','^')[j],s=50,label=label if seed==17 else None)
            ax.set(title=title,xlabel='Request time (% of same-session native)',ylabel='Accuracy (%)',ylim=(90.5,96.5))
            ax.legend(loc='lower left',frameon=False,fontsize=8)
        finish(plt,fig,8,'Point estimates only. Filled: seed 17; open: seed 29. Only seed-17 half-scene removal after readout correction passes both fixed checks.')
    print('Rendered six data-only PNG figures in figures/rendered/. No source media read.')


if __name__ == '__main__':
    main()
