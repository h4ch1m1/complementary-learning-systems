# Rogers et al. (2004): PyTorch reconstruction

## Follow-up tuning (2026-10-07)

See `outputs/rogers2004/tuning/REPORT.md` for measured comparisons. Original
`paper_settings` remains untouched. `tune_model.py` compares continuing the
original margin with a tighter .01 margin and a revised encyclopedic prototype.
The latter restores four fruit-shared properties visible in Figure 3, while
keeping exact predicates and all other input draws unchanged. These are
continued-training experiments, not independent initializations.

`refine_stability.py` explicitly extends the model's training procedure to 56
ticks and hard 0/1 cross-entropy targets with a .01 error dead zone. It scores
ticks 21–56, normalizing the total loss weight to the original eight scored
ticks. This changes both the temporal objective and target convention; any
improvement cannot be assigned to one modification alone. It is an extension,
not a claim that these were the 2004 training settings.

The checkpoint loader now reads the saved `environment.npz`, so revised input
data survive reloading. Regression tests check this and compare both new
gradient variants with PyTorch autograd. Selection uses intact performance;
lesion validation uses fresh seed 1707, without refitting to those outcomes.

The high-learning-rate `stable_long` attempt failed and was manually stopped
after a recorded audit; its requested 800 epochs were not completed. The next
extension, `refine_adam.py`, starts again from the successful `stable` checkpoint.
It uses full-batch Adam (lr=.001), mean BCE over ticks 21–56, and a 1e-6
mean-squared-weight regularizer. General-name feature targets are the mean of
eligible exemplars, reducing sampling noise; BCE is linear in targets, so this
matches the expected BCE of sampled binary targets before regularization.
It does not reproduce the earlier margin-gated stochastic objective exactly.
The architecture and lesion/scoring rules are unchanged. Its stopping condition
is all unambiguous outputs within .05 and all inputs settled, or 1000 updates.
No clinical curve is used for optimization or stopping.

Run from repository root:

```sh
python notebooks/rogers2004/tune_model.py --variant continue --epochs 400
python notebooks/rogers2004/tune_model.py --variant tight --epochs 400
python notebooks/rogers2004/tune_model.py --variant data --epochs 400
python notebooks/rogers2004/refine_stability.py --source data --epochs 200
python notebooks/rogers2004/refine_stability.py --source stable --output stable_long --epochs 800 --lr 0.005
python notebooks/rogers2004/refine_adam.py
python notebooks/rogers2004/validate_tuning.py
python notebooks/rogers2004/report_tuning.py
```

These commands reproduce the recorded continuation chain; do not substitute a
different input environment when loading weights. Selection and evaluation
records in the output folders document the actual completed run.

## Scope and evidence

This implements and runs the recurrent semantic network and analogues of all four task families: naming, sorting, word-picture matching, and drawing/delayed copying. It is **not a bitwise reproduction or a quantitative replication of the published curves**. The original 2004 random patterns, initial weights, patient-level measurements and simulator settings are not supplied in the paper. Some input probabilities must be reconstructed. Outputs contain only actual runs of the code.

Sources:

- Rogers et al. (2004), *Psychological Review*, 111, 205–235, DOI: https://doi.org/10.1037/0033-295X.111.1.205. User-supplied `RogersETAL04.pdf`.
- Author lab's RBP handbook: https://web.stanford.edu/group/pdplab/pdphandbookV3/handbookch9.html
- Author lab's later reference implementation: https://web.stanford.edu/group/pdplab/pdptool/pdptool.zip, files `rbp/rogers.net`, `rbp/rogers.m`, `rbp/features.pat`, `rbp_run.m`. Inspected, not executed or copied into this repository.

The later teaching example is not identical to the 2004 methods: its general name assignments and general-name targets differ; `rogers.m` explicitly changes the reported 0.005 learning rate to 0.001 because its simulator did not converge at 0.005. Therefore the teaching patterns are not passed off as the original study's data.

## Model and training

- 64 hidden semantic units; 64 visual units; 40 name units; 112 verbal descriptors (64 perceptual, 32 functional, 16 encyclopedic). 280 units total.
- Independent directed weights from each visible unit to hidden units, from hidden units to visible units, and recurrently within hidden units. No visible-to-visible weights. Hidden self-connections allowed, matching the reference network configuration. Bias fixed at -2, never trained or lesioned.
- Logistic activation, synchronous net-input averaging with dt=0.25. At t=0 unclamped units start at sigmoid(-2)=0.1192. The paper says approximately 0.19, which does not match the stated logistic bias; the reference code uses the logistic value.
- Clamp the selected input pool for states 0–11; release at state 12. Train for 28 ticks (7 intervals), score states 21–28 (last 2 intervals).
- BCE margin targets 0.05/0.95; zero direct error once an output is within the margin. This loss and margin are choices supported by the later reference implementation, not fully specified in the 2004 paper.
- SGD, online updates, no momentum; paper-setting run uses lr=0.005 and 400 epochs. Per-update multiplicative weight decay 0.001/144. Random weights U(-0.125, 0.125), a later-reference setting not supplied in the article.
- BPTT is implemented in PyTorch tensor operations and checked against PyTorch autograd in float64. TorchScript reduces the Python overhead; its deprecation warning in the installed PyTorch version does not affect the current run. `gradient_scale=1` uses the actual derivative of the dt-scaled loss. The later PDPTool code omits a dt factor in its final weight derivative, equivalent here to `gradient_scale=4`; do not silently conflate those learning-rate conventions.
- Every epoch has 144 trials: 48 names, 48 descriptions, 48 visual patterns. General-name cues sample a category/domain member afresh and retain the general name as the name target. All name units, including inactive names, are clamped on a name trial.

## Input reconstruction and choices

The fixed random environment has six categories with eight items each. Shared prototype properties have probability .8, eligible distinctive properties .2, excluded properties 0. Visual and verbal vectors are unique for each item. Data are generated independently of the lesion results, with a recorded random seed. No seed is selected to obtain the desired lesion curves.

The broad visual structure follows pp. 212–213: animal shared/category/idiosyncratic blocks, fewer shared artifact features, four vehicle and four tool prototype features. Fruits follow the prose: five prototype properties, including one general-artifact and one tool property. Figure 3 appears to depict a different fruit allocation; this ambiguity is retained here as an explicit limitation.

Verbal blocks approximate the displayed prototype structure; they are not claimed to be an exact transcription of every symbol in Figure 3. To keep the required 112 units while providing the exact category/domain predicates specified on p. 213, the last eight encyclopedic descriptor slots are reserved for six category labels plus living and man-made labels. Their exact positions and replacement of those slots are reconstruction choices, and can affect results, especially fruit sorting. Inspect `make_environment()` and the saved `environment.npz`.

Naming follows the paper's counts: three broadly named items and five uniquely named items in birds, mammals, vehicles and tools; eight uniquely named items each for household objects and fruits. General name `animal` applies to all 16 birds/mammals at input, `bird` to all birds, and similarly for vehicle/tool. Unlike the later teaching dataset this gives 40 distinct names and 36 uniquely named items.

## Lesions and tasks

- For each damage level, start from intact weights and independently zero each permitted connection with probability p. No inverted-dropout scaling, retraining, weight noise, or cumulative lesions. Masks are deterministic given the recorded lesion seed. Default 50 masks per nonzero damage level (the paper used 100 for naming and 50 for many other tasks); controls need only one identical zero-damage evaluation.
- At test, present 12 states of input then release; allow up to 2000 ticks, stopping after four consecutive updates whose maximum change is below 1e-5. Save residuals and nonconvergent input counts for each lesion. Finite-time outputs are not automatically guaranteed fixed points. An initial fixed-200-tick run showed residual activity; its summary is retained as a sensitivity comparison, while final figures use the extended evaluation. Training-history audits use 120 ticks and are labelled separately. The original training source snapshot matches the training hash; extended evaluation has its own source hash.
- Naming: visual cue; strongest name above .5; otherwise no response. Score unique non-fruit items, distinguish correct / superordinate / within-domain / cross-domain / omissions.
- Sorting: compare exact broad predicates; compare fine predicates within the true broad domain, reflecting the paper's separate category tasks. Always report fruit and other items separately. This is a readout decision, not a trained clinical response model.
- Word-picture matching: nearest hidden state by Euclidean distance. Include all eligible pairs with close, distant or unrelated foils. Average within animal/artifact domains before averaging them; ties receive .5, not an arbitrary index advantage.
- Drawing: name cue to visual threshold .5. Delayed copying: visual cue, then release. Compare with the intact network's output under the same task. Exclude fruits; drawing uses unique names; copying uses all other objects as in the paper. Also report intact outputs against actual training targets so an already-impaired baseline is visible.
- Feature types: own-category majority (>50%), and for domain-shared features a majority in every other eligible category of the domain. The paper's example uses two contrasting categories; with three artifact categories the all-other-category extension is an explicit choice. Feature-type rates are normalized per item and averaged only when the denominator is nonzero; zero denominators are omitted, not treated as perfect scores.
- Immediate copying is not presented as a successful prediction: hard-clamping the supplied visual pattern guarantees it by construction, as the paper itself notes.

## Interpretation

The model's task scores are deterministic consequences of its inputs, training and readout rules. Pattern similarity is partly built into its environment. Qualitative agreement is not proof of the brain mechanism. Disagreement must remain visible. Matching the published 400-epoch schedule does not imply meeting its stated within-0.05 intact criterion. General-name trials have varying targets, so a deterministic output cannot match every individual member simultaneously; this is separated in the convergence audit.

Patient data are not available here and are not recreated from memory. Plots show the new simulations only, with paper figure references for comparison. Error bars quantify variability of lesions to one trained network, not variation across patients or independently trained networks. The report lists measured outcomes beside the published qualitative expectations without changing parameters to force agreement.

## Run

From the repository root, with the existing requirements installed:

```sh
python -m unittest discover -s notebooks/rogers2004 -p "test_*.py" -v
python notebooks/rogers2004/rogers_model.py --epochs 400 --lr 0.005 --trials 50
python notebooks/rogers2004/plot_results.py outputs/rogers2004/paper_settings
```

Use `--evaluate-only --output PATH` to repeat lesion tests without training, or use a different output directory for a changed seed/setting. Existing files in that specified run directory will be replaced. The notebook provides a self-contained copy of the implementation for inspection and execution; regenerate it with `build_notebook.py` after source changes.

Saved files: model checkpoint, exact environment arrays, training history, configs/seeds/version/source hash, intact audit, raw lesion CSV, summary JSON, PNG/SVG figures and an automatically generated measured-results report. No web page content or Introduction is published by these scripts.
