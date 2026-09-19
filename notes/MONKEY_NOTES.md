# Monkey discrimination reconstruction

Completed run (200 paired simulated subjects, seeds 0–199): at nominal weeks 1, 3, 7, 11, 15, lesion means are 0.59325, 0.67100, 0.71975, 0.70425, 0.67750; control expected means are approximately 0.8365, 0.8244, 0.7883, 0.7279, 0.6877. The qualitative recent-memory deficit and remote convergence appear, but the new lesion curve peaks around week 7, whereas the paper's fitted curve is flatter. Presurgical cortical presentations average about 22 to 70 from recent to remote sets, compared with approximately 25 to 63 reported in the paper. We retained the stated main parameters without tuning away these discrepancies.

Source: McClelland, McNaughton & O'Reilly (1995), Psychological Review 102, pp. 442–444, Figure 13. Local source: ../tmp/pdfs/McClelland_McNaughton_OReilly_1995_CLS.pdf. This is an independent implementation, not recovered original code. No external pretrained model or image dataset is used.

## Reported settings

- Cortical network: 50 inputs, 15 hidden units, one output. Two independent 25-bit object codes with P(bit=1)=0.2. Random balanced-in-expectation choice of correct object; output >0.5 selects the first.
- 100 discriminations, five groups of 20, at nominal learning-to-surgery delays of 1, 3, 7, 11 and 15 weeks. Ten daily blocks of two pairs, each repeated 14 times, plus a final exposure to each pair: 15 traces per pair.
- 250 background items, each sampled daily with probability 0.2; 100 days of background pretraining; background continues after surgery.
- Learning rate 0.03, hippocampal decay 0.025/day, daily offline reinstatement multiplier 0.1, test retrieval multiplier 0.07. Two-week postoperative interval, 200 simulated subjects per condition.
- Real experiment summarized by the paper: 11 lesioned and 7 sham-operated monkeys. Lesions included entorhinal and parahippocampal cortex as well as hippocampus. No raw individual-animal records are available in this project.

## Explicit reconstruction choices

- Sigmoid hidden/output units, half squared error, online SGD without momentum, binary targets, uniform [-0.5, 0.5] weights/biases, float64. These details are not all specified in the monkey subsection; they are implementation assumptions, not asserted original values.
- Background items use the same sparse independent input/random target distribution as experimental pairs; their exact original distribution was not specified.
- Surgery is day 0; acquisition starts at -109, -81, -53, -25 and -11, ends nine days later, with final exposures on start+10. Actual acquisition midpoints are 104.5, 76.5, 48.5, 20.5 and 6.5 days before surgery, plotted at nominal weeks. Pretraining days -209 through -110; postoperative updates days 0 through 13; test at day 14. This resolves unspecified day boundaries while retaining the stated training schedule. The prose's 109-day experimental description is interpreted as the presurgical span; the two-week recovery is additional.
- Traces have strength exp(-0.025 * age), following the decay framework. Traces enter after their direct training day; first eligible replay is next day. Replay does not refresh trace strength.
- Background and replay trials are shuffled together, then direct experiences are presented in their reported 14-trial blocks. Exact ordering relative to other trial types was not specified.
- Both groups start with the same 200 presurgical networks (paired simulation), then receive independent postoperative background samples. Each group retains 200 independent simulated subjects. Batched parameters only parallelize subjects; gradients sum independent subject losses, with no division by cohort size or minibatch averaging over trials.
- Control performance integrates independent hippocampal retrieval analytically: H = 1 - product(1 - 0.07 * trace_strength); expected correct = H + (1-H) * cortical_correct. This removes retrieval Monte Carlo noise, not neural-training variability. Error bars are standard errors across simulated subjects of this expected performance, not animal-data uncertainty.

## Outputs and validation

Run `python test_monkey.py`, then `python build_monkey.py`. The builder runs 200 subjects, saves individual scores to outputs/monkey/results.json, plots comparison.png, and creates a self-contained monkey_discrimination.ipynb. The notebook needs NumPy, PyTorch and Matplotlib and can run without local data assets.

The first two panels use approximate hand-read Figure 13a/b mean coordinates. Figure 13a now also includes approximate SEM endpoints read visually from the source; both are explicitly labeled approximate. They are not raw data and are not used to optimize any parameter. The third panel is the actual new run, without curve smoothing. Differences from the original fitted simulation must be reported rather than hidden.

## Object visualization

The learning stage automatically presents all five displayed pairs, 14 direct exposures per pair, then enters testing. One small button switches modes; restarting learning replays from the beginning, and switching to testing cancels pending playback. `Cohort.train` records output before and after real SGD updates without changing gradients or training order. The curve reveals completed trials only, using p for target 1 and 1-p for target 0: target-directed output, not accuracy or a calibrated probability. The display uses subject 1 consistently, keeping reward identities stable for the human learner. The test uses full-schedule outputs including the fifteenth direct exposure, replay and postoperative experience.

`publish_monkey.py` publishes the first pair from each chronological group (indices 80, 60, 40, 20, 0) for every simulated subject, without selecting examples for outcome; the current simplified UI shows subject 1. Ten Wikimedia photos are illustrative identity labels only. All activities, input bits and targets come from training. The opponent menu selects control/lesion and reveals weeks on hover, keyboard focus or touch. Week settings choose different groups, not a longitudinal trace of one pair. Photo sources, authors and license text are in assets/monkey_objects/sources.json and per-image JSON files.

Control demonstrations sample one Bernoulli retrieval with the analytical probability; success uses the memorized target, otherwise the cortical threshold choice. The visitor may choose first or directly reveal the model answer. Human and model badges appear together on reveal; repeated reveal clicks retain the same result until a new human selection, opponent selection or mode transition resets the trial. Group averages remain analytical expected accuracy; demonstration draws do not alter them. Website scripts contain no retraining or pretrained image recognition. `verify_monkey_auto.cjs` tests automatic completion with a virtual browser clock, nested menus, simultaneous badges, no-human results, stable repeated reveals, timer cancellation and mobile overflow.

Tests compare stacked online updates with ordinary nn.Sequential + autograd to 1e-12, including unequal trial counts, and verify all 100 items have exactly 15 direct experiences. The objective is the qualitative consolidation/forgetting pattern; a close numerical reproduction is not guaranteed by the reported settings alone.
