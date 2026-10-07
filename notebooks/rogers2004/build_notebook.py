"""Generate a self-contained notebook from the tested implementation."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
model_source = (HERE/'rogers_model.py').read_text(encoding='utf8').split("if __name__ == '__main__':")[0]
model_source = model_source.replace('ROOT = Path(__file__).resolve().parents[2]',
    'ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "website").is_dir() and (p / "notebooks").is_dir()), Path.cwd())')
model_source = model_source.replace("hashlib.sha256(Path(__file__).read_bytes()).hexdigest()", "'embedded_notebook_source'")
plot_source = (HERE/'plot_results.py').read_text(encoding='utf8').split("if __name__ == '__main__':")[0]
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(display_name='Python 3',language='python',name='python3')
nb.cells = [
    nbf.v4.new_markdown_cell('''# Rogers et al. (2004): semantic memory and deterioration

**A PyTorch mechanism reconstruction, not an exact recovery of the published curves.**

The recurrent network learns mappings among names, verbal descriptors and visual features. After training, connections are removed without retraining; the same damaged model is tested in naming, sorting, word-picture matching and drawing/delayed copying.

The 2004 original pattern realization, weights and patient-level records are unavailable here. New seeded inputs follow the published structure, with explicit reconstruction choices. No patient measurements are invented. Read `notes/ROGERS2004_NOTES.md` for the full audit.

Sources: [paper](https://doi.org/10.1037/0033-295X.111.1.205), [author lab RBP documentation](https://web.stanford.edu/group/pdplab/pdphandbookV3/handbookch9.html).

The next cell contains the complete implementation so this notebook can run independently. Dependencies: `torch`, `numpy`, `matplotlib`, `scipy`. CPU execution; no GPU required.'''),
    nbf.v4.new_code_cell(model_source),
    nbf.v4.new_markdown_cell('''## 1. Check the reconstructed environment

48 items, 6 categories, 40 names; 216 visible and 64 hidden units. Similarity structure is partly supplied by the input design. Category/domain predicate locations are a reconstruction choice, not recovered original values.'''),
    nbf.v4.new_code_cell('''torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
env = make_environment(2004)
print('Targets:', env['targets'].shape)
print('Names:', len(env['names']))
print('Unique visual patterns:', np.unique(env['targets'][:, 152:], axis=0).shape[0])
print('Permitted connections:', int(connection_mask().sum()))'''),
    nbf.v4.new_markdown_cell('''## 2. Train and audit the intact network

The supplied run uses 400 epochs and learning rate .005, online SGD, fixed bias -2 and 4 ticks per interval. BPTT and continuous-time details follow the documented implementation choices; not all were specified in the article. Do not equate task accuracy with every output satisfying the paper's 0.05 criterion.

`RUN_TRAINING=False` loads the supplied checkpoint when available. Set it to `True` to train from scratch. A standalone copy without the checkpoint automatically trains. Each run saves the environment, parameters, random seeds, raw measurements and model weights.'''),
    nbf.v4.new_code_cell('''OUTPUT = ROOT / 'outputs' / 'rogers2004' / 'paper_settings'
RUN_TRAINING = False
if RUN_TRAINING or not (OUTPUT / 'model.pt').exists():
    model, env = train(seed=12345, data_seed=2004, epochs=400, lr=.005,
                       gradient_scale=1., output=OUTPUT)
else:
    model, env = load(OUTPUT)
print('Training-time audit at 120 ticks:')
print(json.dumps(intact_metrics(model, env), indent=2))'''),
    nbf.v4.new_markdown_cell('''## 3. Lesion the same model and run all four tasks

Each lesion starts from intact weights. Connections are independently removed, without dropout rescaling. 50 masks per nonzero severity; this is fewer than the paper's 100 naming masks. General labels are ambiguous and are excluded where a unique object name is required.

Evaluation allows up to 2000 ticks, stopping after four updates smaller than 1e-5, and records residual changes and nonconvergent input counts. A capped test must not be described as a fixed point if it is still changing. Immediate copying is not a model prediction, because visual input is clamped to the answer.'''),
    nbf.v4.new_code_cell('''if RUN_TRAINING or not (OUTPUT / 'summary.json').exists():
    summary = evaluate(model, env, OUTPUT, trials=50, lesion_seed=707)
else:
    summary = json.loads((OUTPUT / 'summary.json').read_text())
for key in ['naming/all/correct', 'matching/close', 'matching/distant', 'matching/unrelated']:
    print(key, summary['0.2'][key])'''),
    nbf.v4.new_markdown_cell('''## 4. Plot measured results

Bands are SEM over lesion masks for **one** trained network, not uncertainty over patients or independently trained networks. These panels contain no digitized paper data. Differences from the paper should remain visible; do not select seeds or fit lesion curves to hide them.'''),
    nbf.v4.new_code_cell(plot_source),
    nbf.v4.new_code_cell('''from IPython.display import display, Image, Markdown
plot_results(OUTPUT)
display(Image(filename=str(OUTPUT / 'training.png')))
display(Image(filename=str(OUTPUT / 'lesion_results.png')))
display(Image(filename=str(OUTPUT / 'representations.png')))'''),
    nbf.v4.new_markdown_cell('''## 5. Compare predictions with actual outcomes

The generated report contains actual values beside qualitative expectations, including nonmatches. A successful simulation establishes a possible mechanism under these assumptions, not its uniqueness or biological truth.'''),
    nbf.v4.new_code_cell("display(Markdown((OUTPUT / 'RESULTS.md').read_text(encoding='utf8').split('![Training]')[0]))"),
]
nbf.write(nb, HERE/'semantic_deterioration.ipynb')
print(HERE/'semantic_deterioration.ipynb')
