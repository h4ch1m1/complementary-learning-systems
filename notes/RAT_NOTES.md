# Rat consolidation reconstruction

Source: McClelland, McNaughton & O'Reilly (1995), Psychological Review 102, pp. 441–442, Figure 12. Local source: `../tmp/pdfs/McClelland_McNaughton_OReilly_1995_CLS.pdf`.

## Reported protocol

- Cortical network: 16 inputs, 16 hidden units, 16 outputs.
- Twenty random binary stimulus–response background associations, then one new ETS association.
- Each background association occurs once per simulated day; ETS replay is interleaved with background learning. Hippocampal trace decay is ignored in this simulation.
- Cortical performance is measured by reduction of ETS output MSE divided by its pre-ETS value.
- The authors adjusted the learning rate to approximate the animal data. Their original random patterns and exact rat-specific fitted settings are not available here.
- Animals were lesioned at 1, 7, 14 or 28 days and tested seven days after surgery. The paper's simulation curve tracks cortical acquisition; the default reconstruction follows that curve, not an explicit seven-day postsurgical protocol.

## Explicit reconstruction choices

Independent Bernoulli(0.5) inputs and output bits; targets 0.05/0.95; sigmoid hidden/output layers; biases; uniform initialization [-0.5, 0.5]; online SGD with half-summed squared error and no momentum; 100 background pretraining epochs; learning rate 0.3; one ETS replay per day; randomized daily order; float64 CPU tensors.

These initial settings are illustrative, not a recovered parameter set. Pretraining duration and learning rate affect consolidation speed. Early NumPy exploratory runs informed the illustrative settings. The separate Figure 12 workflow below now fits a learning rate to explicitly labeled approximate plot readings.

The optional `lesion_day` intervention stops ETS replay after that day's updates and continues background learning. This is an additional mechanism check. No cortical neurons are removed, no hippocampal neural network is implemented, and no sham-control freezing response is synthesized.

Ten fixed seeds (0–9) show variability. The shaded range is min–max, not a confidence interval. Scores are not clipped: negative values indicate increased error relative to the initial baseline.

## Run

Use the project's `.venv` kernel and run `rat_consolidation.ipynb` from top to bottom. `rat_core.py` contains the same model functions; `build_rat_notebook.py` copies their definitions into the notebook for reading. The existing website is unchanged.

## Figure 12 comparison

`rat_figure12_reproduction.ipynb` is a self-contained Colab-ready reproduction notebook. It includes data, PyTorch model, fitting, plots, sensitivity checks, and a day slider (when ipywidgets is available). No repository upload or Drive access is required. CPU is sufficient for this small model; Colab Pro/GPU is optional, not required.

Data: `data/rat_figure12b.csv`, with extraction status and source recorded in `data/rat_figure12b_provenance.md`. These are approximate normalized means and plotted bar endpoints, NOT original individual-animal data. Primary-source lookup did not locate an analysis-ready raw dataset. No error-bar type was inferred. Values are rounded to 0.01.

Fit: 25 learning rates from 0.05 to 0.65, 16 fixed seeds (0–15) per candidate, minimum unweighted squared error across the four behavioral means; pretraining fixed at 100 epochs. Selected learning rate: 0.35. New-seed audit: 64 fixed seeds (100–163), no parameter refitting. These are new simulation seeds, NOT held-out animal observations. The mean audit predictions are approximately 0.116, 0.396, 0.525, 0.655 at days 1,7,14,28, with RMSE 0.074 against the approximate means. Three of four predictions fall inside the plotted error bars; day 1 is slightly above the upper endpoint. This does not exactly recover the original all-error-bars claim. Do not select a favorable individual seed or call this independent validation of the biological mechanism.

Sensitivity: 0,50,100,200,500 background pretraining epochs at the same fitted learning rate, seeds 100–115. This is a diagnostic, not another tuning stage. The batch implementation uses independent parameters and SGD updates per network; parity against the readable single-network implementation was checked to 1e-12. The shaded band in the comparison plot is the empirical 10th–90th percentile across networks, not a confidence interval.

Run `python rat_fit.py` for machine-readable outputs, or upload `rat_figure12_reproduction.ipynb` to Colab and run all cells. Results are written under `outputs/rat/`; the website is unchanged. Image/video learning remains a separate future extension.
