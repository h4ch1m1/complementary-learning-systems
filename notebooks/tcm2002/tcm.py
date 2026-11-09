"""Howard & Kahana (2002), TCM: explicit associative learning in PyTorch.

Matrix superscripts name the direction: M_FT maps F -> T, M_TF maps T -> F.
No gradient descent, semantic similarity, primacy gain, or recall stopping rule.
See README.md for the equation (9) normalization issue and simulation choices.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Parameters:
    beta: float = 0.402
    gamma: float = 1.0
    tau: float = 0.247
    distractor_duration: float = 7.24

    def __post_init__(self):
        if not 0 < self.beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be positive and finite")
        if not math.isfinite(self.distractor_duration) or self.distractor_duration < 0:
            raise ValueError("distractor_duration must be finite and nonnegative")


class TemporalContextModel(nn.Module):
    """Stateful, single-list TCM with orthonormal item representations.

    One-hot items are coordinates in an orthonormal F basis, not semantic
    embeddings. Initial item -> context columns are mutually orthogonal unit
    vectors. Extra T dimensions hold initial context and distractor inputs.
    All tensors are registered buffers; learning is equations (1), (9), (12).
    """

    def __init__(self, n_items: int, context_dim: int | None = None,
                 params: Parameters | None = None, *, dtype=torch.float64,
                 device="cpu"):
        super().__init__()
        if n_items < 1:
            raise ValueError("n_items must be positive")
        context_dim = context_dim or (2 * n_items + 2)
        if context_dim < n_items + 1:
            raise ValueError("context_dim needs item directions plus initial context")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("use float32 or float64")
        self.params = params or Parameters()
        self.n_items = n_items
        initial = torch.zeros(context_dim, n_items, dtype=dtype, device=device)
        initial[:n_items] = torch.eye(n_items, dtype=dtype, device=device)
        self.register_buffer("M_FT", initial)
        self.register_buffer("M_TF", torch.zeros(n_items, context_dim,
                                                dtype=dtype, device=device))
        context = torch.zeros(context_dim, dtype=dtype, device=device)
        context[n_items] = 1
        self.register_buffer("context", context)

    def clone(self):
        return copy.deepcopy(self)

    def retrieve_context(self, item: int) -> Tensor:
        """Eq. (5): M^FT f, specialized to orthonormal one-hot f."""
        if not 0 <= item < self.n_items:
            raise IndexError(item)
        return self.M_FT[:, item].clone()

    @staticmethod
    def evolution(previous: Tensor, incoming: Tensor, beta: float) -> tuple[Tensor, Tensor]:
        """Eqs. (6)-(7); both nonzero states must have unit norm.

        The zero-input case is explicitly an unchanged context (footnote 8).
        We do not silently renormalize incoming vectors to conceal bad learning.
        """
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if previous.shape != incoming.shape or previous.ndim != 1:
            raise ValueError("context vectors must have equal one-dimensional shapes")
        tol = 1e-5 if previous.dtype == torch.float32 else 1e-10
        if not torch.isclose(previous.norm(), previous.new_tensor(1.), atol=tol, rtol=tol):
            raise ValueError("previous context is not unit length")
        if incoming.count_nonzero() == 0:
            return previous.clone(), previous.new_tensor(1.)
        if not torch.isclose(incoming.norm(), incoming.new_tensor(1.), atol=tol, rtol=tol):
            raise ValueError("retrieved input is not unit length; check Eq. (9)")
        dot = torch.dot(previous, incoming)
        rho = torch.sqrt(1 + beta**2 * (dot.square() - 1)) - beta * dot
        return rho * previous + beta * incoming, rho

    @torch.no_grad()
    def integrate(self, incoming: Tensor) -> Tensor:
        updated, rho = self.evolution(self.context, incoming, self.params.beta)
        self.context.copy_(updated)
        return rho

    @torch.no_grad()
    def present(self, item: int, *, learn: bool = True) -> dict[str, Tensor]:
        """Section 3.2: retrieve -> integrate -> update BOTH association matrices.

        This is also the sequence on successful recall. Suppression/editing,
        when requested by a caller, occurs downstream of the core model.
        """
        incoming = self.retrieve_context(item)
        rho = self.integrate(incoming)
        gamma = self.params.gamma
        # Normalization constraint in 3.1.2 implies the SQUARE ROOT below.
        B = torch.rsqrt(incoming.new_tensor(gamma**2 + 1)
                        + 2 * gamma * torch.dot(incoming, self.context))
        A = gamma * B
        if learn:
            # Eq. (12): only the column parallel to f changes.
            self.M_FT[:, item].copy_(A * incoming + B * self.context)
            # Eq. (1): store the POST-integration contextual state.
            self.M_TF[item].add_(self.context)
        return {"incoming": incoming, "context": self.context.clone(),
                "rho": rho, "A": A, "B": B}

    def activations(self, cue: Tensor | None = None) -> Tensor:
        """Eq. (2), with one-hot f_i."""
        return self.M_TF @ (self.context if cue is None else cue)

    def probabilities(self, excluded=()) -> Tensor:
        """Eq. (14): softmax(2*a/tau), over the current list only.

        Default excludes nothing. Passing excluded removes already-recalled
        items from the choice set, so the CRP is conditional on a transition
        to a not-yet-recalled item, as in the standard lag-CRP analysis.
        """
        logits = 2 * self.activations() / self.params.tau
        if len(excluded):
            logits = logits.clone()
            logits[list(excluded)] = -torch.inf
        if not torch.isfinite(logits).any():
            raise ValueError("all items are excluded")
        return torch.softmax(logits, dim=0)

    @torch.no_grad()
    def distract(self, duration: float, direction: Tensor):
        """Effective distractor interval, a documented simulation convention.

        r=sqrt(1-beta^2); t' = r^d*t + sqrt(1-r^(2d))*u, with u orthogonal
        to all preceding inputs. For integer d this is exactly a block of d
        novel orthogonal inputs; for real d it interpolates that decay law.
        It represents information input, not passive elapsed clock time.
        No distractor-to-list association is stored.
        """
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        if duration == 0:
            return
        tol = 1e-5 if self.context.dtype == torch.float32 else 1e-10
        if direction.shape != self.context.shape or not torch.isclose(
                direction.norm(), direction.new_tensor(1.), atol=tol, rtol=tol):
            raise ValueError("distractor direction must be a unit context vector")
        if torch.dot(direction, self.context).abs() > tol or torch.max(
                (self.M_TF @ direction).abs()) > tol or torch.max(
                (self.M_FT.T @ direction).abs()) > tol:
            raise ValueError("use a fresh distractor direction orthogonal to stored history")
        retention = (1 - self.params.beta**2)**(duration / 2)
        self.context.mul_(retention).add_(math.sqrt(1 - retention**2) * direction)


CONDITIONS = ("immediate", "delayed", "continuous")


def study_list(condition: str, params: Parameters | None = None,
               n_items: int = 12, *, device="cpu") -> tuple[TemporalContextModel, Tensor]:
    """Section 4.1 list of 12 distinct items; return test state and study contexts.

    immediate:  f1 f2 ... f12 | test
    delayed:    f1 f2 ... f12 RI | test
    continuous: f1 D f2 D ... f12 D RI | test

    In the continuous-distractor paradigm every item, including the last,
    is followed by its interpresentation distractor; the retention interval
    then follows. The last item is therefore two intervals (2d) from test.
    This structure reproduces the model points of Figs. 7C and 8B.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    model = TemporalContextModel(n_items, params=params, device=device)
    history = []
    next_direction = n_items + 1

    def interval():
        nonlocal next_direction
        direction = torch.zeros_like(model.context)
        direction[next_direction] = 1
        model.distract(model.params.distractor_duration, direction)
        next_direction += 1

    for item in range(n_items):
        history.append(model.present(item)["context"])
        if condition == "continuous":
            interval()
    if condition != "immediate":
        interval()
    return model, torch.stack(history)


def study_sequence(events: str, params: Parameters | None = None, *, device="cpu"):
    """Study an arbitrary sequence such as "f D f D f D D".

    Each "f" presents the next list item; each "D" is one distractor interval
    of effective length d. Returns the test state and the study contexts.
    """
    tokens = events.split()
    if not tokens or set(tokens) - {"f", "D"} or "f" not in tokens:
        raise ValueError('events must contain "f" and optional "D" tokens')
    n_items, n_intervals = tokens.count("f"), tokens.count("D")
    model = TemporalContextModel(n_items, context_dim=n_items + 1 + max(n_intervals, 1),
                                 params=params, device=device)
    history, item, direction = [], 0, n_items + 1
    for token in tokens:
        if token == "f":
            history.append(model.present(item)["context"])
            item += 1
        else:
            unit = torch.zeros_like(model.context)
            unit[direction] = 1
            model.distract(model.params.distractor_duration, unit)
            direction += 1
    return model, torch.stack(history)


def transition_distribution(model: TemporalContextModel, *, suppress_repeats=False):
    """Exact probabilities for first response and second response conditional on first.

    Branch on every possible first response, then execute the SAME Eq. (5)-(12)
    update as on study. Useful as a noise-free check on 3000-trial simulation.
    """
    first = model.probabilities()
    second = []
    for item in range(model.n_items):
        branch = model.clone()
        branch.present(item)
        second.append(branch.probabilities([item] if suppress_repeats else ()))
    return first, torch.stack(second)


def expected_lag_crp(first: Tensor, second: Tensor):
    """Opportunity-corrected CRP for first -> second response only.

    numerator(l) = sum_i P(first=i)*P(second=i+l | first=i).
    denominator(l) = sum_i P(first=i) * 1[0 <= i+l < L].
    Zero lag is omitted. Under raw Eq. (14), repetitions consume probability
    mass but are not a nonzero-lag transition; no silent renormalization.
    """
    n = first.numel()
    lags = torch.tensor([x for x in range(1-n, n) if x], device=first.device)
    numerator, denominator = [], []
    for lag in lags.tolist():
        src = torch.arange(max(0, -lag), min(n, n-lag), device=first.device)
        numerator.append((first[src] * second[src, src+lag]).sum())
        denominator.append(first[src].sum())
    num, den = torch.stack(numerator), torch.stack(denominator)
    return lags, num / den, num, den


def sample_first_two(first: Tensor, second: Tensor, trials=3000, seed=2002):
    """Vectorized sampling from the enumerated branches; not a different model."""
    if trials < 1:
        raise ValueError("trials must be positive")
    generator = torch.Generator(device=first.device).manual_seed(seed)
    a = torch.multinomial(first, trials, replacement=True, generator=generator)
    b = torch.multinomial(second[a], 1, generator=generator).squeeze(1)
    return torch.stack((a, b), dim=1)


def empirical_first_two(responses: Tensor, n_items: int):
    """Responses are ZERO-based positions, shape (trials, 2); repeats permitted."""
    if responses.ndim != 2 or responses.shape[1] != 2 or responses.shape[0] < 1:
        raise ValueError("responses must have nonempty shape (trials, 2)")
    if responses.dtype != torch.long or responses.min() < 0 or responses.max() >= n_items:
        raise ValueError("responses must contain valid zero-based int64 item indices")
    a, b = responses.T
    first = torch.bincount(a, minlength=n_items).double() / len(a)
    lags = torch.tensor([x for x in range(1-n_items, n_items) if x], device=a.device)
    num = torch.stack([((b-a) == lag).sum() for lag in lags])
    den = torch.stack([((a+lag >= 0) & (a+lag < n_items)).sum() for lag in lags])
    crp = num.double() / den.clamp_min(1)
    crp[den == 0] = torch.nan
    return first, lags, crp, num, den
