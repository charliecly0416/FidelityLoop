"""Render the registered N5 panels from accepted CPU source tables (no GPU)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

REGIMES = ('steady', 'recovery', 'burst_offline')
TITLES = ('Steady', 'Recovery', 'Burst + offline')
POLICIES = ('all1', 'all2', 'hysteresis_u8_d0_c60')
LABELS = ('All1', 'All2', 'Dynamic')
COLORS = ('#4C78A8', '#8E9AA6', '#009E73')
MODELS = ('original', 'calibrated')
MODEL_COLORS = ('#D55E00', '#0072B2')
STATES = dict(off='#E5E5E5', starting='#E69F00', active='#009E73',
              draining='#56B4E9', stopping='#CC79A7')


def read_csv(root, name):
    with (root / (name + '.csv')).open() as f:
        return list(csv.DictReader(f))


def subset(rows, **keys):
    return [r for r in rows if all(r[k] == str(v) for k, v in keys.items())]


def points(ax, x, values, color, marker='o'):
    """All three repeats, mean, and descriptive min--max (never a CI)."""
    if len(values) != 3:
        raise ValueError('exactly three repeat points required')
    avg = statistics.mean(values)
    ax.plot([x, x], [min(values), max(values)], color=color, lw=.8, zorder=5)
    ax.plot([x - .055, x + .055], [avg, avg], color=color, lw=1.4, zorder=6)
    ax.scatter(x + np.array([-.055, 0, .055]), values, s=12, marker=marker,
               facecolors='white', edgecolors=color, linewidths=.7, zorder=7)


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.5, length=2)


def export(fig, output, name, qa):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    undrawn_ticks = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    undrawn_ticks.update((id(tick.label1), id(tick.label2)))
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text() or id(item) in undrawn_ticks:
            continue
        # Matplotlib retains invisible/out-of-range tick labels as Text objects.
        if item.axes is not None and item.get_clip_on():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -2 or box.y0 < -2 or
                box.x1 > fig.bbox.width + 2 or box.y1 > fig.bbox.height + 2):
            outside.append(item.get_text())
    if outside:
        raise ValueError(f'{name}: text outside canvas: {outside}')
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(output / f'{name}.{suffix}', dpi=300, facecolor='white')
    qa.append(dict(figure=name,canvas_text_bounds='PASS',outside_text=outside,
                   width_mm=round(fig.get_figwidth()*25.4,2),
                   height_mm=round(fig.get_figheight()*25.4,2)))
    plt.close(fig)


def cost_figure(data, output, qa):
    rows = read_csv(data, 'run_metrics')
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.0), layout='constrained',
                            gridspec_kw=dict(height_ratios=[1.25, 1, .85]))
    components = ('window_holding_cost', 'transition', 'api_accept_cost',
                  'offline_deadline_miss_penalty')
    component_colors = ('#4C78A8', '#9D79BC', '#E69F00', '#D55E00')
    for j, regime in enumerate(REGIMES):
        w = 'locked_test_' + regime + '_v2'
        for k, policy in enumerate(POLICIES):
            rs = subset(rows, window=w, policy=policy)
            base = 0
            for component, color in zip(components, component_colors):
                values = [(.002*int(r['startup_events_in_window']) +
                           .001*int(r['shutdown_events_in_window'])) if component == 'transition'
                          else float(r[component]) for r in rs]
                v = statistics.mean(values)
                axs[0,j].bar(k, v, bottom=base, width=.62, color=color, linewidth=0)
                base += v
            points(axs[0,j], k, [float(r['window_total_cost']) for r in rs], '#222222')
            points(axs[1,j], k, [float(r['gpu_occupied_seconds']) for r in rs], COLORS[k])
            points(axs[2,j], k, [int(r['offline_on_time']) for r in rs], COLORS[k])
        axs[0,j].set_title(TITLES[j], fontsize=8, fontweight='bold')
        axs[0,j].set_ylim(0, 4.65)
        axs[1,j].set_ylim(0, 4500)
        axs[2,j].set_ylim(112.8, 121.1)
        axs[2,j].set_yticks([114, 116, 118, 120])
        axs[2,j].axhline(114, color='#D55E00', lw=.7, ls='--')
        for i in range(3):
            axs[i,j].set_xticks(range(3), LABELS if i == 2 else [])
            axs[i,j].set_xlim(-.6, 2.6)
            tidy(axs[i,j])
    for ax, label in zip(axs[:,0], ('Window cost\n(scenario USD)', 'Occupied resource\n(GPU s)',
                                   'Offline on time\n(out of 120)')):
        ax.set_ylabel(label)
    fig.legend([Patch(color=c) for c in component_colors],
               ['GPU holding', 'Transitions', 'Synthetic API', 'Miss penalty'],
               loc='outside upper center', ncol=4, frameon=False)
    export(fig, output, 'cost_slo', qa)


def prediction_figure(data, output, qa):
    rows = read_csv(data, 'prediction_errors')
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.5), layout='constrained')
    metrics = ('shared_local_service_mae_seconds', 'gpu_occupied_seconds_absolute_error',
               'gpu_active_seconds_absolute_error')
    for j, policy in enumerate(POLICIES):
        for k, regime in enumerate(REGIMES):
            w = 'locked_test_' + regime + '_v2'
            for m, model in enumerate(MODELS):
                rs = subset(rows, window=w, policy=policy, model=model)
                for i, metric in enumerate(metrics):
                    points(axs[i,j], k + (m-.5)*.27, [float(r[metric]) for r in rs],
                           MODEL_COLORS[m], ('o','s')[m])
        axs[0,j].set_title(LABELS[j], fontweight='bold', fontsize=8)
        axs[0,j].set_yscale('log')
        axs[0,j].set_ylim(.009, 10)
        for i in (1,2):
            axs[i,j].set_ylim((-.3,6) if j<2 else (-25,1100 if i==1 else 820))
        for i in range(3):
            axs[i,j].set_xticks(range(3), ['Steady','Recovery','Burst'] if i == 2 else [])
            axs[i,j].set_xlim(-.5,2.5)
            tidy(axs[i,j])
    for ax,label in zip(axs[:,0], ('Shared-local MAE (s)\nlog scale', 'Occupation error\n(absolute GPU s)',
                                  'Active error\n(absolute GPU s)')):
        ax.set_ylabel(label)
    fig.legend([Line2D([],[],color=c,marker=m,linestyle='none',mfc='white')
                for c,m in zip(MODEL_COLORS,('o','s'))], ['Original', 'Calibrated'],
               loc='outside upper center', ncol=2, frameon=False)
    export(fig, output, 'prediction_error', qa)


def timeline_figure(data, output, qa):
    timeline = read_csv(data, 'timeline_repeat1')
    states = read_csv(data, 'state_transitions_repeat1')
    misses = read_csv(data, 'offline_misses_repeat1')
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 5.8), layout='constrained',
                            gridspec_kw=dict(height_ratios=[1.2, .9, 1]))
    for j, regime in enumerate(REGIMES):
        w = 'locked_test_' + regime + '_v2'
        rs = subset(timeline, window=w)
        time = np.array([int(r['tick'])/60 for r in rs])
        for field, color, style in (('arrivals','#777777','-'),('queue_online','#0072B2','-'),
                                    ('queue_offline','#D55E00','-')):
            axs[0,j].plot(time,[int(r[field]) for r in rs],lw=.65,color=color,ls=style,label=field)
        axs[0,j].set_title(TITLES[j] + ' / repeat 1', fontsize=8, fontweight='bold')
        for field,color,style in (('predicted_target','#D55E00','--'),('target','#0072B2','-')):
            axs[1,j].step(time,[int(r[field]) for r in rs],where='post',lw=.8,color=color,ls=style)
        axs[1,j].set_yticks([0,1,2]);axs[1,j].set_ylim(-.1,2.2)
        for gpu in (0,1):
            changes = sorted(subset(states, window=w, gpu=gpu), key=lambda r: float(r['time_s']))
            for k,r in enumerate(changes):
                start = max(0,float(r['time_s']))
                end = min(2100,float(changes[k+1]['time_s'])) if k+1<len(changes) else 2100
                if end > start:
                    axs[2,j].broken_barh([(start/60,(end-start)/60)],(gpu-.28,.56),
                                          facecolors=STATES[r['state']],linewidths=0)
        api = [int(r['tick'])/60 for r in rs if int(r['api_accepts'])]
        axs[2,j].scatter(api,[-.7]*len(api),marker='|',color='#222222',s=12,lw=.6)
        for r in subset(misses, window=w):
            axs[2,j].scatter(float(r['deadline_s'])/60,1.6,marker='x',s=18,color='#D55E00')
        axs[2,j].set_yticks([-.7,0,1,1.6],['API','GPU0','GPU1','Miss'])
        axs[2,j].set_ylim(-1,1.95)
        axs[2,j].set_xlabel('Time (min)')
        for i in range(3):
            axs[i,j].set_xlim(0,35)
            axs[i,j].set_xticks([0,10,20,30,35])
            if i<2:axs[i,j].set_xticklabels([])
            tidy(axs[i,j])
    axs[0,0].set_ylabel('Requests / queue')
    axs[1,0].set_ylabel('Target instances')
    lines = [Line2D([],[],color=c,lw=.9,ls=s) for c,s in
             [('#777777','-'),('#0072B2','-'),('#D55E00','-'),('#0072B2','-'),('#D55E00','--')]]
    fig.legend(lines,['Arrivals / s','Online queue','Offline queue','Actual target','Predicted target'],
               loc='outside upper center', ncol=3, frameon=False)
    fig.legend([Patch(color=c) for c in STATES.values()],list(STATES),
               loc='outside lower center', ncol=5, frameon=False)
    export(fig, output, 'lifecycle_repeat1', qa)


def price_figure(data, output, qa):
    rows = read_csv(data, 'price_group_statistics')
    fig, axs = plt.subplots(1,3,figsize=(7.2,2.65),layout='constrained')
    prices = (.5,1.,2.)
    for j,regime in enumerate(REGIMES):
        w = 'locked_test_' + regime + '_v2'
        grid = np.array([[100*float(subset(rows,window=w,baseline='all1',
                           gpu_multiplier=g,api_multiplier=a)[0]['savings_ratio_of_means'])
                          for a in prices] for g in prices])
        im=axs[j].imshow(grid,cmap='BrBG',vmin=-80,vmax=80,origin='lower',aspect='equal')
        for (y,x),v in np.ndenumerate(grid):
            axs[j].text(x,y,f'{v:+.1f}%',ha='center',va='center',fontsize=7,
                        color='white' if abs(v)>48 else '#222222')
        axs[j].set_xticks(range(3),['0.5','1','2'])
        axs[j].set_yticks(range(3),['0.5','1','2'])
        axs[j].set_xlabel('API price multiplier')
        axs[j].set_title(TITLES[j],fontsize=8,fontweight='bold')
        axs[j].tick_params(length=0)
    axs[0].set_ylabel('GPU price multiplier')
    cb=fig.colorbar(im,ax=axs,location='bottom',shrink=.6,aspect=40,pad=.04)
    cb.set_label('Total window cost saved vs All1 (%)')
    export(fig, output, 'price_sensitivity', qa)


def render(data, output):
    data,output=Path(data),Path(output)
    manifest=json.loads((data/'analysis_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('source data digest mismatch: '+name)
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
                         'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
                         'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,
                         'figure.facecolor':'white','savefig.facecolor':'white',
                         'svg.hashsalt':'n5-20260920'})
    qa=[]
    for plot in (cost_figure,prediction_figure,timeline_figure,price_figure):
        plot(data,output,qa)
    report=dict(status='AUTOMATED_EXPORT_PASS',figures=qa,python=sys.version,
                matplotlib=matplotlib.__version__,numpy=np.__version__,
                source_manifest_sha256=hashlib.sha256((data/'analysis_manifest.json').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                statistical_unit='run: three repeats per policy and window; range is not a CI',
                visual_review='separate record; not implied by automated export',
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())})
    (output/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],figures=qa),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.data,args.output)
