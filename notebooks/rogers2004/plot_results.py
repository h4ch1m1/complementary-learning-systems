"""Plot measured simulation outputs only; never substitutes published curves."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def plot_results(output):
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    train = json.loads((output/'training.json').read_text())
    metrics = json.loads((output/'intact_metrics.json').read_text())
    config = json.loads((output/'config.json').read_text())
    levels = sorted(float(x) for x in summary)
    def series(key):
        return np.array([summary[str(x)][key]['mean'] for x in levels])
    def line(ax, key, label, color, style='-'):
        y = series(key)
        sem = np.array([summary[str(x)][key]['sem'] for x in levels])
        ax.plot(np.array(levels)*100, y, style, color=color, label=label, lw=2)
        ax.fill_between(np.array(levels)*100, y-sem, y+sem, color=color, alpha=.12)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.facecolor':'white'})
    colors = ['#276a9c','#d17832','#52916c','#986ba3','#687580']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax = axes[0,0]
    for key, label, color in zip(['correct','omission','semantic','superordinate','cross_domain'],
                                ['Correct','No response','Related name','General name','Cross-domain'],colors):
        line(ax,f'naming/all/{key}',label,color)
    ax.set_title('A  Picture naming (cf. Fig. 6)')
    ax = axes[0,1]
    for mod, color in [('picture',colors[0]),('word',colors[1])]:
        for level, style in [('general','-'),('specific','--')]:
            line(ax,f'sorting/{mod}/animal_artifact/{level}',f'{mod.title()}: {level}',color,style)
    ax.set_title('B  Sorting animals / artifacts (cf. Fig. 8)')
    ax = axes[0,2]
    for mod, color in [('picture',colors[0]),('word',colors[1])]:
        for level, style in [('general','-'),('specific','--')]:
            line(ax,f'sorting/{mod}/fruit/{level}',f'{mod.title()}: {level}',color,style)
    ax.set_title('C  Fruit sorting (cf. Fig. 9)')
    ax = axes[1,0]
    for key,color in zip(['close','distant','unrelated'],colors):
        line(ax,f'matching/{key}',key.title()+' foil',color)
    ax.axhline(.5,color='#aaa',lw=1,ls=':')
    ax.set_title('D  Word-picture matching (cf. Fig. 10)')
    ax = axes[1,1]
    for task,color in [('drawing',colors[0]),('delayed_copy',colors[1])]:
        for error,style in [('omission','-'),('intrusion','--')]:
            line(ax,f'{task}/all/{error}',task.replace('_',' ').title()+': '+error,color,style)
    ax.set_title('E  Visual feature errors (cf. Figs. 12-13)')
    ax = axes[1,2]
    for key,label,color in zip(['shared_domain','shared_category','distinctive'],
                                ['Domain-shared','Category-shared','Distinctive'],colors):
        line(ax,f'drawing/{key}/omission',label,color)
    ax.set_title('F  Drawing omissions by feature (cf. Fig. 14)')
    for ax in axes.flat:
        ax.set(xlabel='Connections removed (%)',ylabel='Proportion',ylim=(-.03,1.03))
        ax.grid(alpha=.15); ax.legend(fontsize=8,loc='best',frameon=False)
    fig.suptitle('Rogers et al. (2004) — PyTorch mechanism reconstruction\n'
        f'Seed {config["seed"]}; reconstructed inputs; bands = SEM across lesion masks; no patient data',fontsize=14)
    fig.savefig(output/'lesion_results.png',dpi=180)
    fig.savefig(output/'lesion_results.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot([r['epoch'] for r in train],[r['train_mse'] for r in train],color=colors[0])
    axes[0].set(xlabel='Training epoch',ylabel='Mean squared error',title='Training (includes stochastic broad-name trials)')
    for key,label,color in [('naming_accuracy','Naming',colors[0]),('feature_bit_accuracy','Feature bits',colors[1]),('within_005_fraction','Within 0.05 of target',colors[2])]:
        axes[1].plot([r['epoch'] for r in train],[r[key] for r in train],label=label,color=color)
    axes[1].set(xlabel='Training epoch',ylabel='Proportion correct',ylim=(-.03,1.03),title='Training audit at 120 ticks (not final settling)')
    axes[1].legend(frameon=False)
    fig.savefig(output/'training.png',dpi=180); plt.close(fig)
    env = np.load(output/'environment.npz')
    out = np.load(output/'intact_activations.npz')['outputs']
    labels = env['item_labels'].tolist()
    fig, axes = plt.subplots(1,3,figsize=(15,9),constrained_layout=True)
    for ax, matrix, title in zip(axes,[env['targets'][:,152:],env['targets'][:,40:152],out[2,:,216:]],
                                 ['Visual input','Verbal descriptions','Learned semantic states']):
        dendrogram(linkage(matrix,method='average',metric='euclidean'),labels=labels,
                   orientation='right',leaf_font_size=7,ax=ax,color_threshold=0,above_threshold_color=colors[0])
        ax.set_title(title); ax.set_xlabel('Euclidean distance (average linkage)')
    fig.suptitle('Representational structure — reconstructed inputs, not digitized original figures')
    fig.savefig(output/'representations.png',dpi=160); plt.close(fig)

    checks = []
    def check(label, actual, expected): checks.append((label,actual,expected))
    m20 = summary['0.2']
    val=lambda k:m20[k]['mean']
    check('Naming at 20% lesion',f'{val("naming/all/correct"):.3f}', 'Declines with damage')
    check('Picture sorting: general minus specific (20%)',f'{val("sorting/picture/animal_artifact/general")-val("sorting/picture/animal_artifact/specific"):+.3f}', 'Positive')
    check('Fruit picture sorting: specific minus general (20%)',f'{val("sorting/picture/fruit/specific")-val("sorting/picture/fruit/general"):+.3f}', 'Positive (reversal)')
    check('Matching: close / distant / unrelated (20%)', ' / '.join(f'{val("matching/"+k):.3f}' for k in ['close','distant','unrelated']), 'Increasing')
    check('Drawing: distinctive minus domain-shared omissions (20%)',f'{val("drawing/distinctive/omission")-val("drawing/shared_domain/omission"):+.3f}', 'Positive')
    check('Drawing minus delayed-copy total errors (20%)',f'{val("drawing/all/total_errors")-val("delayed_copy/all/total_errors"):+.3f}', 'Positive; eligible items differ as in paper')
    max_residual=max(summary[str(x)]['settle_residual']['mean'] for x in levels)
    with (output/'lesion_trials.csv').open(encoding='utf8') as stream:
        raw = list(csv.DictReader(stream))
    unsettled = sum(int(row['nonconvergent_inputs']) > 0 for row in raw)
    unsettled_visual = sum(int(row['nonconvergent_visual']) > 0 for row in raw)
    report = ['# Measured reconstruction results','',
        '**This is a mechanism reconstruction with newly generated feature vectors, not an exact reproduction of the published curves.**', '',
        f'Training: seed={config["seed"]}, data seed={config["data_seed"]}, epochs={config["epochs"]}, learning rate={config["lr"]}, gradient scale={config["gradient_scale"]}.', '',
        '## Intact network', '',
        *[f'- {k}: {v:.6f}' for k,v in metrics.items()], '',
        f'Intact exact-naming criterion: {"PASS" if metrics["naming_accuracy"] == 1 else "NOT MET"}. All audited outputs within 0.05: {"PASS" if metrics["max_error"] < .05 else "NOT MET"}. Qualitative lesion effects below do not override these failures.', '',
        'The within-0.05 audit excludes ambiguous general-name cues, whose targets vary by trial. It includes all visual and verbal cues. This fraction must not be confused with all units passing the paper criterion.', '',
        '## Paper predictions compared with this run', '',
        '| Measure | Actual simulation | Published qualitative expectation |','|---|---|---|',
        *[f'| {a} | {b} | {c} |' for a,b,c in checks], '',
        f'Maximum mean final-tick residual over lesion levels: {max_residual:.6g}. Evaluation allows up to 2000 ticks and stops after four consecutive updates smaller than 1e-5. Raw CSV includes nonconvergent input counts; capped trials must not be described as proven fixed points.', '',
        f'**Convergence limitation:** {unsettled}/{len(raw)} trials reached the cap with at least one unsettled input; {unsettled_visual} had unsettled visual inputs. These trials remain in the plots (not silently discarded), so the curves are finite-time responses where convergence failed. The intact network has {metrics["nonconvergent_inputs"]} unsettled inputs.', '',
        'Error bands are standard errors over lesion masks for one trained network. They are not patient uncertainty or between-training-seed uncertainty. No patient measurements were fabricated, digitized, or fitted.', '',
        '![Training](training.png)','![Lesions](lesion_results.png)','![Representations](representations.png)']
    (output/'RESULTS.md').write_text('\n'.join(report),encoding='utf8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    plot_results(parser.parse_args().output)
