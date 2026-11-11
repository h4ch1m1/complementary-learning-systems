"""Shared, paper-aligned tutorial plots. Run this file to rebuild website PNGs.

Only measured model outputs are plotted. Experimental readings are labeled;
changing presentation does not fit parameters or replace simulation results.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/paper_figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.labelsize': 12, 'axes.titlesize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'savefig.bbox': None})
BLUE, ORANGE, GREEN = '#0173b2', '#de8f05', '#029e73'


def panel(width=4.6, height=3.9):
    return plt.subplots(figsize=(width, height))


def semantic_clusters(snapshots, indices, labels):
    from scipy.cluster.hierarchy import linkage, dendrogram
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=True)
    for ax, epoch in zip(axes, [25, 200, 500]):
        tree = linkage(np.asarray(snapshots[epoch])[indices], method='average')
        dendrogram(tree, labels=labels, leaf_rotation=90, color_threshold=0,
                   above_threshold_color='black', ax=ax, leaf_font_size=10)
        ax.set(xlabel=f'Epoch {epoch}', ylim=(0, max(1.5, tree[-1, 2] * 1.08)))
    axes[0].set_ylabel('Euclidean Distance')
    return fig


def semantic_error(losses):
    fig, ax = panel(5.2, 3.5)
    ax.plot(range(len(losses)), losses, color='black')
    ax.set(xlabel='Epoch', ylabel='Mean squared error', xlim=(0, len(losses)-1))
    return fig


def animal_readings(ax, data):
    ax.errorbar(data.day, data['mean'],
                yerr=[data['mean']-data.lower, data.upper-data['mean']],
                fmt='s-', mfc='white', color=ORANGE, ms=4, capsize=4,
                label='K & F Data (approx.)')


def rat_performance(mean, data):
    fig, ax = panel()
    ax.plot(range(len(mean)), mean, color=BLUE, label='Simulation')
    animal_readings(ax, data)
    ax.set(xlabel='Consolidation Period in Days', ylabel='Proportion of Maximum Performance',
           xlim=(-2, 30), ylim=(-.05, 1), title='Lesioned Animals’ Performance')
    ax.legend(fontsize=9)
    return fig


def rat_sensitivity(curves, data):
    fig, ax = panel(5.2, 3.5)
    for i, (duration, values) in enumerate(curves.items()):
        ax.plot(range(len(values)), values, color=plt.cm.viridis(.12+.8*i/4), label=f'{duration} epochs')
    animal_readings(ax, data)
    ax.set(xlabel='Consolidation Period in Days', ylabel='Proportion of Maximum Performance',
           ylim=(0, 1), title='Background pretraining: sensitivity')
    ax.legend(fontsize=9)
    return fig


def monkey_retention(result):
    fig, ax = panel()
    for key, marker, color in [('lesion', 'o', ORANGE), ('control', 's', BLUE)]:
        values = np.asarray(result[key])
        ax.errorbar(result['weeks'], values.mean(0), yerr=values.std(0, ddof=1)/np.sqrt(len(values)),
                    marker=marker, mfc='white', color=color, label=key.title(), capsize=3)
    ax.set(xticks=result['weeks'], ylim=(.5, 1), xlabel='Learning–Lesion Interval (Weeks)',
           ylabel='Proportion Correct', title=f'Tutorial model: {result["subjects"]} subjects/group')
    ax.legend(fontsize=10)
    return fig


def two_system(study, key, ax=None):
    fig, ax = panel() if ax is None else (ax.figure, ax)
    for condition, label, color in [('control', 'Control', BLUE), ('lesion', 'Hippocampal lesion', '#d55e00')]:
        values = study['curves'][condition]
        ax.plot(range(len(values)), values, color=color, label=label)
    ylabel = {'winocur': 'Percent Sample Eaten', 'kim': 'Percent Freezing',
              'monkey': 'Percent Correct', 'squire': 'Predicted recall probability'}[key]
    limits = {'winocur':(.5,.9), 'kim':(0,.8), 'monkey':(.5,.9), 'squire':(0,1)}[key]
    ax.set(xlabel='Learning–Lesion Interval (Days)', ylabel=ylabel, title=study['name'],
           ylim=limits)
    ax.legend(fontsize=9)
    return fig


def rogers_summary():
    """Means and mask SEMs for the published 10,000-update trial table."""
    import pandas as pd
    trials = pd.read_csv(ROOT/'data/rogers2004/lesion_trials.csv').drop(columns='trial')
    grouped = trials.groupby('lesion')
    means, sems, counts = grouped.mean(), grouped.sem().fillna(0), grouped.size()
    return {str(float(level)): {key: {'mean':float(row[key]), 'sem':float(sems.loc[level,key]),
                                     'n':int(counts.loc[level])} for key in means.columns}
            for level, row in means.iterrows()}


def rogers_panels(summary):
    levels = sorted(map(float, summary))
    def values(key, selected=levels):
        return [summary[str(level)][key]['mean'] for level in selected]
    figures = {}
    fig, ax = panel(4.6, 4.2)
    for key, label, color, style in [('semantic', 'Semantic errors', GREEN, '-'),
            ('superordinate', 'Superordinate errors', ORANGE, '--'),
            ('cross_domain', 'Cross-domain errors', '#cc78bc', ':'), ('omission', 'Omissions', '#d55e00', '-.')]:
        ax.plot(values('naming/all/correct'), values('naming/all/'+key), marker='o', mfc='white',
                ms=4, color=color, ls=style, label=label)
    ax.set(xlabel='Proportion correct', ylabel='Proportion of responses by error type',
           xlim=(1, 0), ylim=(0, 1), title='Model: full measured severity range')
    figures['rogers_naming'] = fig
    fig, ax = panel(4.6, 4.2)
    for key, color in [('close', BLUE), ('distant', ORANGE), ('unrelated', GREEN)]:
        ax.plot(levels, values('matching/'+key), 'o-', mfc='white', ms=4, color=color,
                label=key.title()+' foil'+(' — Fig. 10 curve' if key=='close' else ' — tutorial extension'))
    ax.set(xlabel='p connections lesioned', ylabel='Proportion correct', xlim=(0, .5),
           ylim=(.5, 1.02), title='Model: word–picture matching')
    figures['rogers_matching'] = fig
    fig, ax = panel(4.6, 4.2)
    selected = [x for x in levels if x <= .35]
    for key, marker, color, style in [('general', 'o', BLUE, '-'), ('specific', '^', '#d55e00', '--')]:
        ax.plot(selected, values('sorting/picture/animal_artifact/'+key, selected),
                marker=marker, ms=4, color=color, ls=style, label=key.title())
    ax.set(xlabel='p connections lesioned', ylabel='Pictures: proportion correct',
           xlim=(-.01, .36), ylim=(.4, 1.02), title='Model')
    ax.grid(axis='y', ls=':', color='.5', lw=.65)
    figures['rogers_sorting'] = fig
    for name, groups, keys, labels, xlabel, ylabel, title in [
        ('rogers_omissions', [.2, .25, .3], ['drawing/'+x+'/omission' for x in ['shared_domain', 'shared_category', 'distinctive']],
         ['SDom', 'SCat', 'Dist'], 'Feature type', 'Proportion omitted', 'Drawing'),
        ('rogers_fruit', ['general', 'specific'], [.1, .2, .3, .4], ['.1', '.2', '.3', '.4'],
         'p connections lesioned', 'Fruits (p correct)', 'Model: Pictures')]:
        fig, ax = panel(4.6, 3.8); width=.8/len(groups)
        for i, group in enumerate(groups):
            y = [values(key, [group])[0] for key in keys] if name=='rogers_omissions' else values('sorting/picture/fruit/'+group, keys)
            label=f'{group:.2f} Conn. les.' if name=='rogers_omissions' else group.title()
            ax.bar(np.arange(len(keys))+(i-(len(groups)-1)/2)*width, y, width,
                   color=['black', '.65', 'white'][i], edgecolor='black', label=label)
        ax.set(xticks=range(len(keys)), xticklabels=labels, xlabel=xlabel, ylabel=ylabel, title=title, ylim=(0, 1.05))
        if name=='rogers_fruit': ax.axhline(.5, color='black', ls=':', lw=1)
        figures[name] = fig
    fig, ax = panel(4.6, 3.5)
    ax.errorbar(levels, values('nonconvergent_inputs'),
                yerr=[summary[str(x)]['nonconvergent_inputs']['sem'] for x in levels], color='black', capsize=3)
    ax.set(xlabel='Proportion of connections lesioned', ylabel='Mean unsettled input probes', title='Settling: 10,000-update cap')
    figures['rogers_settling'] = fig
    for fig in list(figures.values())[:-1]: fig.axes[0].legend(fontsize=9)
    return figures


def tcm_recall(pfr, lags, crp, condition):
    """Exact model expectations; only delayed recall has digitized observations."""
    import pandas as pd
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    observed = pd.read_csv(ROOT/'data/tcm2002/experimental_delayed.csv')
    for ax, measure in zip(axes, ['pfr', 'crp']):
        x, y = (np.arange(1, 13), np.asarray(pfr)) if measure=='pfr' else (np.asarray(lags), np.asarray(crp))
        keep = np.ones(len(x), bool) if measure=='pfr' else abs(x)<=5
        groups = [keep] if measure=='pfr' else [keep & (x<0), keep & (x>0)]
        for i, select in enumerate(groups):
            ax.plot(x[select], y[select], 'o-', color=BLUE, mfc='white', ms=4, label='Tutorial model' if i==0 else None)
        if condition=='delayed':
            points = observed.loc[observed.measure==measure]
            for i, group in enumerate([points] if measure=='pfr' else [points.loc[points.x<0], points.loc[points.x>0]]):
                ax.errorbar(group.x, group.y, yerr=[group.y-group.lower, group.upper-group.y],
                            fmt='o-', color='black', ms=4, capsize=3, label='Experimental (digitized)' if i==0 else None)
        ceiling = .6 if measure=='pfr' else .45
        ax.set(xlabel='Serial Position' if measure=='pfr' else 'Lag',
               ylabel='Probability of First Recall' if measure=='pfr' else 'Conditional Response Probability',
               ylim=(0, ceiling if max(y[keep])<=ceiling else 1))
        ax.legend(fontsize=9)
    return fig


def rebuild():
    """Rebuild static website figures from the accepted, retained measurements."""
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    figures = {}
    archive = np.load(ROOT/'outputs/representations_seed0.npz')
    targets = pd.read_csv(ROOT/'data/targets.csv', index_col=0)
    concepts = list(dict.fromkeys(x.split(' | ')[0] for x in targets.index))
    labels = ['oak','pine','rose','daisy','canary','robin','salmon','sunfish']
    figures['semantic_clusters'] = semantic_clusters({e:archive[f'epoch_{e}'] for e in [25,200,500]}, [concepts.index(x) for x in labels], labels)
    figures['semantic_error'] = semantic_error(pd.read_csv(ROOT/'outputs/learning_history.csv').mse)
    rat = np.load(ROOT/'outputs/rat/figure12_curves.npz')
    data = pd.read_csv(ROOT/'outputs/rat/figure12_comparison.csv')
    figures['rat_performance'] = rat_performance(rat['audit'].mean(0), data)
    sensitivity = pd.read_csv(ROOT/'data/cls1995/rat_sensitivity.csv')
    figures['rat_sensitivity'] = rat_sensitivity({p:sensitivity[str(p)].to_numpy() for p in [0,50,100,200,500]}, data)
    figures['monkey_retention'] = monkey_retention(json.loads((ROOT/'outputs/monkey/results.json').read_text()))
    for key, study in json.loads((ROOT/'outputs/two_system/results.json').read_text()).items():
        figures['two_'+key] = two_system(study, key)
    summary = rogers_summary()
    figures.update(rogers_panels(summary))
    # Browser widgets consume the same accepted summary and digitized data.
    metrics = ['naming/all/correct', 'matching/close', 'matching/distant', 'matching/unrelated',
               'sorting/picture/animal_artifact/general', 'sorting/picture/animal_artifact/specific']
    browser = {'levels': sorted(map(float, summary)), 'max_settle_ticks': 10000,
               'summary': {level: {key: {'mean': values[key]['mean']} for key in metrics}
                           for level, values in summary.items()}}
    (ROOT/'website/rogers_results.js').write_text('window.rogersResults = '+json.dumps(browser, separators=(',', ':'))+';\n')
    data = pd.read_csv(ROOT/'data/tcm2002/experimental_delayed.csv')
    observed = {'delayed': {measure: group[['x','y','lower','upper']].values.tolist()
                            for measure, group in data.groupby('measure')}}
    (ROOT/'website/tcm_experimental.js').write_text('window.tcmExperimental = '+json.dumps(observed, separators=(',', ':'))+';\n')
    for name, fig in figures.items():
        fig.tight_layout(); fig.savefig(OUT/(name+'.png'), dpi=300, bbox_inches=None); plt.close(fig)
    return figures


if __name__ == '__main__':
    rebuild()
