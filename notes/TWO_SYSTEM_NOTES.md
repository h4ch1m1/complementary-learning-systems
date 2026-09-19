# Simplified two-system model

## Source

McClelland, McNaughton & O'Reilly (1995), published Psychological Review version:
Figure 14 (p. 444), Equations 3–9 and Table 1 (p. 445), task descriptions on pp. 442 and 446.
The draft PDF uses different figure numbers. Use the published version when referring to Figure 1 and Figure 14.

## Implementation

- PyTorch float64, deterministic difference equations. No sampled subjects, optimizer, or new fit.
- One synchronous update per day; both increments use old h and c. This gives h(t)=h(0)*(1-Dh)^t. Do not silently mix this with h(0)*exp(-Dh*t).
- C combines learning rate and replay rate; this model does not identify them separately.
- All five parameters in each preset are copied from Table 1 without tuning.
- Rh=Rc=1 as suggested on p. 445.
- Lesion is applied after the chosen number of intact daily updates. It sets h to 0 immediately; c is preserved at that instant and decays thereafter without consolidation.
- Controls continue consolidation through the recovery interval. Both conditions are tested at lesion delay + recovery.
- Recovery: Winocur 10 days, Kim/Fanselow 7 days, monkey 14 days. Squire/Cohen uses 0 additional days, representing unavailability at retrieval, not an anatomical lesion or a treatment simulation.
- Baseline: 0.5 for food choice and object discrimination; 0 for freezing and free recall. Free recall is plotted on a proportion scale, not the original count axis; no arbitrary item-count multiplier is fitted.
- The interactive shows trajectories and expected performance, not individual sampled choices. Each point in a lesion-delay curve represents a separate hypothetical lesion time.

## Reproduction boundary

Figure 1 combines experimental points and fitted curves. The publication does not supply raw data, fitting code, full numerical timing conventions, or response scaling for all four panels. The curves here reconstruct Equations 3–9 under the explicit choices above; they are not claimed to be an exact reproduction of those fitted lines. No artificial experimental points or standard errors are generated. The original Figure 1 is displayed separately as a reference.

## Verification

`test_two_system.py` checks the first update, discrete exponential decay, range bounds, no-consolidation limit, lesion timing, exact postlesion cortical decay, baseline retrieval, and available PyTorch gradients.
The notebook is executed in the existing project environment. Browser tests compare rendered values with exported PyTorch output, exercise all presets, and check mobile overflow.

## Run

    .venv\Scripts\python.exe -m unittest test_two_system.py
    .venv\Scripts\python.exe build_two_system.py

The builder updates only the equations chapter and its own artifacts.
