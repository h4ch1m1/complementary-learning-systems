"""An explicit training extension for the reconstructed CLS semantic network.

The paper comparison remains the 500-epoch run. This module diagnoses incomplete
item knowledge by freezing the learned concept representations and continuing
online SGD on the downstream readout until every training output has the correct
on/off response. It does not relabel later epochs as 500,
fit the published hierarchy, select a seed, or imply out-of-sample generalization.
"""
from __future__ import annotations

import numpy as np
import torch


def knowledge_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """On/off accuracy uses the task's existing 0.5 decision threshold.

    MSE and maximum absolute error retain the distinction between correct binary
    responses and matching the continuous 0.05/0.95 targets.
    """
    with torch.no_grad():
        errors = prediction - target
        expected = target > .5
        correct = (prediction > .5) == expected
        return {
            'wrong_outputs': int((~correct).sum()),
            'outputs': int(correct.numel()),
            'binary_accuracy': float(correct.double().mean()),
            'positive_recall': float(correct[expected].double().mean()),
            'mse': float(errors.square().mean()),
            'max_absolute_error': float(errors.abs().max()),
        }


def continue_to_completion(model, optimizer, order_rng: np.random.Generator,
                           inputs, relations, targets, *, start_epoch: int = 500,
                           max_epoch: int = 20000, stable_epochs: int = 25,
                           freeze_representation: bool = True) -> dict:
    """Continue online training; stop after sustained correct factual responses.

    By default, concept-to-representation weights/biases are frozen; only the
    downstream hidden/output mapping is refined. This is an explicitly changed
    training protocol, designed to preserve the learned representational result.
    All examples contribute one half-summed-square-error update each epoch, in
    the supplied generator's shuffled order. The caller provides a copy of the
    reference model and generator when preserving the 500-epoch comparison.
    ``max_epoch`` is a cap, not a guaranteed convergence claim. No item/seed is
    excluded from the criterion. The complete 0.05/0.95 tolerance is not the
    stopping criterion; maximum continuous error is reported independently.
    """
    if max_epoch < start_epoch or stable_epochs < 1:
        raise ValueError('Require max_epoch >= start_epoch and stable_epochs >= 1.')
    if freeze_representation:
        for parameter in model.representation.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    history = []
    stable = 0
    for epoch in range(start_epoch, max_epoch + 1):
        if epoch > start_epoch:
            for i in order_rng.permutation(len(inputs)):
                optimizer.zero_grad()
                prediction = model(inputs[i], relations[i])
                loss = .5 * (prediction - targets[i]).square().sum()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            prediction = model(inputs, relations)
            metrics = knowledge_metrics(prediction, targets)
        stable = stable + 1 if metrics['wrong_outputs'] == 0 else 0
        if epoch == start_epoch or epoch % 25 == 0 or stable == stable_epochs:
            history.append({'epoch': epoch, **metrics})
        if stable >= stable_epochs:
            break
    return {
        'start_epoch': start_epoch,
        'final_epoch': epoch,
        'converged': stable >= stable_epochs,
        'stable_epochs_required': stable_epochs,
        'stable_epochs_observed': stable,
        'criterion': 'all training outputs have the correct >0.5 on/off response',
        'representation_frozen': freeze_representation,
        'history': history,
        'metrics': metrics,
        'prediction': prediction.detach().clone(),
    }
