# Displayed lesion results

`lesion_trials.csv` contains the measured columns used by the website figures,
from the retained checkpoint in `outputs/rogers2004/tuning/adam/`. It contains
one intact observation and 50 independent masks at each nonzero lesion level,
using mask seed 707 and a 10,000-update settling cap. No retraining or lesion
rescaling was applied. Naming, matching and sorting columns are proportions;
drawing columns are proportions of eligible features omitted. The plotting
helper calculates means and sample SEMs across masks on this one network.

`settle_residual` is the largest remaining update across input probes;
`nonconvergent_inputs` counts probes not satisfying the convergence tolerance.
Some damaged probes remain unsettled at the cap. Their responses are retained,
so these finite-time results do not establish convergence of every attractor.

The reconstructed feature vectors and optimized training differ from the original
paper. In particular, naming still overproduces commission errors relative to
omissions at severe impairment. The longer evaluation budget is a numerical
improvement, not an exact recovery of the published findings.
