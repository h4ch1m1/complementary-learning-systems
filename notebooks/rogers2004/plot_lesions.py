"""Plot the six measured lesion outcomes from the optimized model."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rogers_model as m

def main():
    root = m.ROOT / 'outputs/rogers2004/tuning'
    summary = json.loads((root/'adam/summary.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    axes=axes.ravel()
    groups=[('Naming',['naming/all/correct','naming/all/semantic','naming/all/omission']),
            ('Word-picture matching',['matching/close','matching/distant','matching/unrelated']),
            ('Sorting pictures',['sorting/picture/animal_artifact/general','sorting/picture/animal_artifact/specific']),
            ('Drawing omissions',['drawing/shared_domain/omission','drawing/shared_category/omission','drawing/distinctive/omission']),
            ('Fruit sorting',['sorting/picture/fruit/general','sorting/picture/fruit/specific']),
            ('Dynamics',['nonconvergent_inputs'])]
    x=sorted(float(k) for k in summary)
    for ax,(title,keys) in zip(axes,groups):
        for key in keys:
            if key not in summary['0.2']: continue
            y=[summary[str(v)][key]['mean'] for v in x]
            label=key.split('/')[-2] if title=='Drawing omissions' else key.split('/')[-1]
            sem=[summary[str(v)][key]['sem'] for v in x]
            ax.errorbar(x,y,yerr=sem,marker='o',label=label,markersize=3,capsize=2)
        ax.set_title(title); ax.set_xlabel('Fraction of connections removed'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root/'lesion_validation.png',dpi=180); plt.close(fig)

if __name__ == '__main__':
    main()
