# Papers

Interactive explanations and computational models that replicate studies on learning and memory.

## Complementary learning systems

McClelland, McNaughton, and O’Reilly (1995) examine how the brain can learn new experiences rapidly while gradually acquiring knowledge that generalizes across experiences. Their account combines rapid learning in the hippocampus with slower learning in the neocortex, using replay to integrate new memories with existing knowledge.

[Read the interactive article](https://h4ch1m1.github.io/complementary-learning-systems/website/semantic_knowledge.html) · [Paper](https://doi.org/10.1037/0033-295X.102.3.419)

## Structure and deterioration of semantic memory

Rogers and colleagues (2004) investigate how names, visual features, and verbal descriptions contribute to semantic knowledge. A recurrent network learns associations among these forms of information through a shared semantic representation. Damage to its connections produces patterns of impaired naming, classification, matching, and drawing that can be compared with semantic dementia.

[Read the interactive article](https://h4ch1m1.github.io/complementary-learning-systems/website/semantic_memory.html) · [Paper](https://doi.org/10.1037/0033-295X.111.1.205) · [Open in Colab](https://colab.research.google.com/github/h4ch1m1/complementary-learning-systems/blob/main/notebooks/rogers2004/tutorial.ipynb)

## Temporal context in free recall

Howard and Kahana (2002) explain why free recall begins near the end of a list and why each recalled word tends to be followed by its neighbor, usually the word that came after it. In their temporal context model, each word is associated with a gradually changing state of context, and recalling a word reinstates part of the context in which it was studied. The same mechanism accounts for recency, contiguity, and the forward asymmetry of recall across short and long delays.

[Read the interactive article](https://h4ch1m1.github.io/complementary-learning-systems/website/temporal_context.html) · [Paper](https://doi.org/10.1006/jmps.2001.1388) · [Open in Colab](https://colab.research.google.com/github/h4ch1m1/complementary-learning-systems/blob/main/notebooks/tcm2002/tutorial.ipynb)

## Validated improvements

The [semantic notebook](notebooks/cls1995/semantic/semantic_knowledge.ipynb) retains the 500-epoch reference and adds an explicit readout-refinement extension: it learns missed binary training facts while preserving the learned concept representations. This changes the training protocol and does not repair the original epoch-200 timing mismatch.

The Rogers evaluator now exposes a consistent 10,000-update settling budget, while retaining residual flags and labeling archived 2,000-update results. A training continuation that passed the intact-output tolerance was rejected because it worsened other lesion contrasts; the tutorial does not load those candidate weights. TCM probability axes expand when needed to retain all computed values; its model equations and predictions are unchanged.

The original checkpoints and saved baseline results remain intact. Public trial data for the displayed Rogers plots are in [data/rogers2004](data/rogers2004); the TCM experimental readings are in [data/tcm2002](data/tcm2002).

### Approved figure presentation

Run `MPLBACKEND=Agg python notebooks/paper_figures.py` to rebuild the website figures and browser summaries from retained model measurements. The notebook plot cells share these functions. The Rogers results use the accepted 10,000-update evaluation; its remaining naming mismatch is documented. TCM’s delayed-condition overlay uses labeled approximate experimental readings in `data/tcm2002/experimental_delayed.csv`.

The rat sensitivity table stores mean normalized-performance curves from 16 networks per pretraining duration (seeds 100–115). The 64-network rat reference curves remain in `outputs/rat/figure12_curves.npz`. Reports, diagnostic runs, and rejected experiments are local authoring material and are not website dependencies.
