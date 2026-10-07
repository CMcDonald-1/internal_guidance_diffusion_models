"""Target densities on {-1,1}^d with EXACT ground truth by enumeration.

Every target here is small enough to enumerate all 2^d states, which buys four
things you cannot get at scale and should exploit while you can:

  * exact i.i.d. sampling from pi   (no MCMC bias in the training set)
  * the exact tilt mean operator    m(h) = sum_x x pi(x) e^<h,x> / Z
  * the exact tilted target q       for end-to-end TV evaluation
  * the exact log partition function

Targets are JSON-serialisable so a dataset can be regenerated or re-evaluated
against the same pi later.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

MAX_ENUM_DIM = 24


def tau_for_target_success_rate(p_target: float, p0: float) -> float:
    """Exact tau solving for a target Bernoulli success rate under a +-1

    linear reward tilt `pi(x, x*) * exp(tau * x*)`, `x* in {-1,+1}`. Given
    the untilted success rate `p0 = P_pi(x*=+1)`, the tilted rate is the
    standard logistic reweighting `p_target = p0*e^tau / (p0*e^tau +
    (1-p0)*e^-tau)`; this inverts that closed-form exactly (no search
    needed) -- generic to any target exposing a reward-augmented coordinate
    (`RewardAugmented`, `SudokuRewardAugmented`, `Sudoku6Target`, ...), not
    specific to any one of them. `lambda_for_target` below is the analogous
    inversion for a `{0,1}`-valued (rather than `{-1,+1}`-valued) tilt.
    """
    return 0.5 * math.log((p_target * (1 - p0)) / (p0 * (1 - p_target)))


def lambda_for_target_success_rate(p_target: float, p0: float) -> float:
    """Exact lambda solving for a target success rate under a 0/1-valued

    reward tilt `pi(x) * exp(lambda * r(x))`, `r(x) in {0,1}` -- see
    `tau_for_target_success_rate`'s docstring for the `{-1,+1}` version.
    """
    return math.log((p_target * (1 - p0)) / (p0 * (1 - p_target)))


def enumerate_cube(d: int, device=None) -> torch.Tensor:
    """All 2^d states of {-1,1}^d, shape (2^d, d)."""
    if d > MAX_ENUM_DIM:
        raise ValueError(f"d={d} too large to enumerate (limit {MAX_ENUM_DIM})")
    bits = (torch.arange(2**d, device=device).unsqueeze(1) >> torch.arange(d, device=device)) & 1
    return 2.0 * bits.float() - 1.0


# --------------------------------------------------------------------- rewards
def r_parity(x: torch.Tensor) -> torch.Tensor:
    return x.prod(dim=-1)


def r_majority(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x.sum(dim=-1) + 1e-9)


def make_r_threshold(k: int):
    """+1 iff at least k coordinates are +1.  Small k => rare success."""

    def r(x: torch.Tensor) -> torch.Tensor:
        return torch.where((x > 0).sum(-1) >= k, 1.0, -1.0)

    return r


def make_r_exact(k: int):
    """+1 iff EXACTLY k coordinates are +1, else -1.  A narrow band, not a tail."""

    def r(x: torch.Tensor) -> torch.Tensor:
        return torch.where((x > 0).sum(-1) == k, 1.0, -1.0)

    return r


REWARDS = {"parity": r_parity, "majority": r_majority}


def get_reward(spec: str):
    if spec.startswith("threshold:"):
        return make_r_threshold(int(spec.split(":")[1]))
    if spec.startswith("exact:"):
        return make_r_exact(int(spec.split(":")[1]))
    if spec == "sudoku_valid":
        # local import: tilt.sudoku imports EnumerableTarget from this module,
        # so importing it at module load time would be circular.
        from . import sudoku as _sudoku

        return _sudoku.r_sudoku_valid
    return REWARDS[spec]


# --------------------------------------------------------------------- targets
class EnumerableTarget:
    """Base class.  Subclasses implement log_weights() and spec()."""

    def __init__(self, d: int, device=None):
        self.d = d
        self.device = device
        self.states = self._build_states(device)
        lw = self.log_weights(self.states)
        self.log_w = lw - lw.max()
        self.log_Z = torch.logsumexp(self.log_w, 0)
        self.probs = torch.exp(self.log_w - self.log_Z)

    # -- to implement --------------------------------------------------
    def log_weights(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _build_states(self, device=None) -> torch.Tensor:
        """Enumerate the support.  Default: all 2^d states (MAX_ENUM_DIM=24).

        Subclasses whose support is a small, structured subset of the full
        hypercube (e.g. permutation-encoded rows, d > MAX_ENUM_DIM) should
        override this to construct their states directly rather than
        enumerating and filtering the full cube.
        """
        return enumerate_cube(self.d, device)

    def spec(self) -> dict:
        raise NotImplementedError

    # -- exact quantities ----------------------------------------------
    def sample(self, n: int, generator=None) -> torch.Tensor:
        idx = torch.multinomial(self.probs, n, replacement=True, generator=generator)
        return self.states[idx]

    def mean(self) -> torch.Tensor:
        return self.probs @ self.states

    def effective_support(self) -> float:
        p = self.probs
        return float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))

    @torch.no_grad()
    def natural_mean(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """m(h) = E_{pi_h}[X] where pi_h(x) ∝ pi(x) exp(<h,x>).  EXACT."""
        out = []
        for i in range(0, h.shape[0], chunk):
            logits = self.log_w.unsqueeze(0) + h[i : i + chunk] @ self.states.T
            out.append(torch.softmax(logits, 1) @ self.states)
        return torch.cat(out)

    def G(self, u: torch.Tensor, a: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """Exact tilt mean operator in VP coordinates."""
        from .noising import beta_of_a

        return self.natural_mean(beta_of_a(a) * u, chunk=chunk)

    @torch.no_grad()
    def log_partition(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """log Z(h) = log E_pi[exp(<h,X>)], the ABSOLUTE (properly normalized)

        tilted log-partition -- not shifted by any arbitrary constant. Needed
        to combine several targets into an exact mixture (see tilt/sudoku.py):
        for a mixture pi_mix = sum_i w_i pi_i, log Z_mix(h) is a logsumexp
        over log(w_i) + log_partition_i(h), and the mixture's tilted mean is
        the corresponding softmax-weighted combination of each component's
        natural_mean(h).
        """
        out = []
        for i in range(0, h.shape[0], chunk):
            logits = self.log_w.unsqueeze(0) + h[i : i + chunk] @ self.states.T
            out.append(torch.logsumexp(logits, dim=1))
        return torch.cat(out) - self.log_Z

    def tilted_probs(self, h: torch.Tensor) -> torch.Tensor:
        """Exact state probabilities of q(x) ∝ pi(x) exp(<h,x>).  h is (d,)."""
        return torch.softmax(self.log_w + self.states @ h, 0)

    def sample_tilted(self, h: torch.Tensor, n: int, generator=None) -> torch.Tensor:
        """Exact i.i.d. samples from q(x) propto pi(x) exp(<h,x>).  h is (d,).

        Default: tilted_probs + explicit enumeration (needs self.states).
        Subclasses whose support can't be materialized (e.g.
        sudoku.SudokuRewardAugmented at row_noise>0) override this with an
        equally exact, enumeration-free sampler. evaluate.py's mode="ood"
        sampling step calls this instead of assuming `.states` exists.
        """
        probs = self.tilted_probs(h)
        idx = torch.multinomial(probs, n, replacement=True, generator=generator)
        return self.states[idx]

    # -- exact conditioning (hard-fixed coordinates, NOT a soft tilt) ----
    @torch.no_grad()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log pi(x) for an explicit batch of candidate states, by exact
        match against self.states.  Generic fallback for any target with
        materialized support (small explicit lists -- up to 331,776 states
        for sudoku's row_noise=0, up to 2^24 for Ising/product).  -inf for
        any x not in the support.  Subclasses with unmaterialized support
        (e.g. SudokuRowsTarget at row_noise>0) override this with their own
        exact, enumeration-free formula.
        """
        x = torch.atleast_2d(x)
        match = (x.unsqueeze(1) == self.states.unsqueeze(0)).all(-1)  # (n, |states|)
        found = match.any(dim=1)
        idx = match.float().argmax(dim=1)
        return torch.where(found, self.log_w[idx] - self.log_Z,
                            torch.full((x.shape[0],), -float("inf")))

    def _coord_mask(self, coords, values) -> torch.Tensor:
        mask = torch.ones(self.states.shape[0], dtype=torch.bool, device=self.states.device)
        for c, v in zip(coords, values):
            sign = 1.0 if float(v) > 0 else -1.0
            mask &= (torch.sign(self.states[:, int(c)]) == sign)
        return mask

    @torch.no_grad()
    def constraint_log_prob(self, coords, values) -> float:
        """log P_self(x_coords = values): the exact marginal log-probability
        this target's OWN current density assigns to fixing a subset of
        coordinates to given signs.  Generic version for materialized
        support.  Used by mixture-style condition_on_coords implementations
        (see sudoku.SudokuMixtureTarget) to Bayes-update mixing weights
        after conditioning on evidence.
        """
        mask = self._coord_mask(coords, values)
        if not mask.any():
            return float("-inf")
        return float(torch.logsumexp(self.log_w[mask], dim=0) - self.log_Z)

    def condition_on_coords(self, coords, values) -> "EnumerableTarget":
        """Exact pi(x | x_coords = values): the TRUE hard-conditioned target,
        not a soft exponential tilt (a linear tilt tau*(e_A+ - e_A-) only
        recovers this in the tau -> infinity limit -- see the write-up
        discussion this implements). Filters states matching the given
        signs and keeps their relative log-weights unchanged: natural_mean/
        log_partition/tilted_probs/sample all renormalise internally via
        softmax/logsumexp, so no explicit renormalisation is needed here.

        Requires materialized .states -- subclasses without one (e.g.
        SudokuRowsTarget at row_noise>0, SudokuMixtureTarget,
        SudokuRewardAugmented at row_noise>0) override this with their own
        exact, enumeration-free construction (row-factorization / mixture
        decomposition), returning an object exposing this same generic
        interface so callers never need to know which branch they're on.
        """
        if self.states is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no materialized states; "
                "override condition_on_coords"
            )
        mask = self._coord_mask(coords, values)
        if not mask.any():
            raise ValueError("given constraint has probability zero under this target")
        # Shallow-copy self (not a bare EnumerableTarget) so the conditioned
        # object keeps its actual subclass identity -- e.g. tilt_direction()
        # for a RewardAugmented/SudokuRewardAugmented, or any other subclass
        # attribute/method that doesn't depend on which states survived
        # filtering (self.s/self.J for Ising, self.base/self.reward_name/self.r
        # for RewardAugmented, etc.) -- only .states/.log_w/.log_Z/.probs
        # actually need to change.
        import copy

        obj = copy.copy(self)
        log_w = self.log_w[mask]
        obj.states = self.states[mask]
        obj.log_w = log_w - log_w.max()
        obj.log_Z = torch.logsumexp(obj.log_w, 0)
        obj.probs = torch.exp(obj.log_w - obj.log_Z)
        return obj

    # -- io -------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.spec(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "EnumerableTarget":
        return from_spec(json.loads(Path(path).read_text()))


class ProductTarget(EnumerableTarget):
    """pi(x) ∝ exp(<s,x>).  Factorises; G_i(u,a) = tanh(s_i + beta u_i)."""

    def __init__(self, s: torch.Tensor, device=None):
        self.s = torch.as_tensor(s, dtype=torch.float32, device=device)
        super().__init__(len(self.s), device)

    def log_weights(self, x):
        return x @ self.s

    def spec(self):
        return {"kind": "product", "s": self.s.tolist()}

    def closed_form_G(self, u, a):
        from .noising import beta_of_a

        return torch.tanh(self.s + beta_of_a(a) * u)

    @classmethod
    def random(cls, d: int, scale: float = 1.0, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        return cls(torch.randn(d, generator=g) * scale)


class IsingTarget(EnumerableTarget):
    """pi(x) ∝ exp(<s,x> + 0.5 x^T J x), J symmetric with zero diagonal."""

    def __init__(self, s, J, device=None):
        self.s = torch.as_tensor(s, dtype=torch.float32, device=device)
        self.J = torch.as_tensor(J, dtype=torch.float32, device=device)
        super().__init__(len(self.s), device)

    def log_weights(self, x):
        return x @ self.s + 0.5 * (x * (x @ self.J)).sum(1)

    def spec(self):
        return {"kind": "ising", "s": self.s.tolist(), "J": self.J.tolist()}

    def mean_field_0(self, u, a):
        """tanh(s + beta u): ignores J.  The baseline the net must beat."""
        from .noising import beta_of_a

        return torch.tanh(self.s + beta_of_a(a) * u)

    @classmethod
    def random(cls, d: int, J0: float = 1.0, field: float = 0.3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        s = torch.randn(d, generator=g) * field
        A = torch.randn(d, d, generator=g) * (J0 / math.sqrt(d))
        J = torch.triu(A, 1)
        return cls(s, J + J.T)


class RewardAugmented(EnumerableTarget):
    """Lift pi on {-1,1}^d to pi(x, x*) on {-1,1}^(d+1) with x* = r(x).

    This is the augmented space from the write-up: the reward becomes an extra
    coordinate, and the reward tilt q ∝ pi e^{tau r(x)} becomes the LINEAR tilt
    q ∝ pi(x,x*) e^{tau x*}, i.e. natural parameter h = tau * e_{d+1}.
    States with x* != r(x) get zero mass.
    """

    def __init__(self, base: EnumerableTarget, reward: str, device=None):
        self.base = base
        self.reward_name = reward
        self.r = get_reward(reward)
        super().__init__(base.d + 1, device)

    def log_weights(self, x):
        core, star = x[:, : self.base.d], x[:, self.base.d]
        lw = self.base.log_weights(core)
        return torch.where(star == self.r(core), lw, torch.full_like(lw, -1e30))

    def spec(self):
        return {"kind": "reward_augmented", "base": self.base.spec(),
                "reward": self.reward_name}

    def tilt_direction(self, tau: float) -> torch.Tensor:
        """Natural parameter h = tau * e_{d+1} for the reward tilt."""
        h = torch.zeros(self.d, device=self.device)
        h[-1] = tau
        return h

    def success_rate(self) -> float:
        """P_pi(r(X) = +1).  Small values are the rare-reward regime."""
        return float(self.probs[self.states[:, -1] > 0].sum())


def from_spec(spec: dict, device=None) -> EnumerableTarget:
    kind = spec["kind"]
    if kind == "product":
        return ProductTarget(spec["s"], device)
    if kind == "ising":
        return IsingTarget(spec["s"], spec["J"], device)
    if kind == "reward_augmented":
        base = from_spec(spec["base"], device)
        if spec["reward"] == "sudoku_valid":
            from . import sudoku as _sudoku

            return _sudoku.SudokuRewardAugmented(base, device)
        return RewardAugmented(base, spec["reward"], device)
    if kind in ("sudoku_rows", "sudoku_valid_solutions", "sudoku_mixture"):
        # local import: tilt.sudoku imports EnumerableTarget from this module,
        # so importing it at module load time would be circular.
        from . import sudoku as _sudoku

        if kind == "sudoku_rows":
            return _sudoku.SudokuRowsTarget(row_noise=spec.get("row_noise", 0.0), device=device)
        if kind == "sudoku_valid_solutions":
            return _sudoku.SudokuValidSolutionsTarget(_sudoku.D, device)
        return _sudoku.SudokuMixtureTarget(
            from_spec(spec["base"], device), spec["p_inject"], device=device
        )
    raise ValueError(f"unknown target kind {kind!r}")


def build_target(kind: str, d: int, seed: int = 0, J0: float = 1.0,
                 field: float = 0.3, reward: str | None = None,
                 row_noise: float = 0.0, p_inject: float | None = None):
    """Convenience constructor used by the CLI scripts.

    row_noise / p_inject only apply to kind="sudoku" (see tilt/sudoku.py):
    row_noise controls SudokuRowsTarget's per-row noise (0.0 = pure
    permutation rows); p_inject, if set, wraps the result in a
    SudokuMixtureTarget that injects extra mass on the 288 valid solutions.
    reward, if set, must be "sudoku_valid" -- the standard RewardAugmented
    route (same mechanism as Ising/product's reward=), lifting to d=33 via
    SudokuRewardAugmented. reward and p_inject now COMPOSE: passing both
    builds SudokuRewardAugmented(SudokuMixtureTarget(SudokuRowsTarget(
    row_noise), p_inject)) -- reward-injection at the base-density level,
    then reward-augmentation on top -- so the reward-augmented (d+1)
    dataset actually contains p_inject-many genuine reward-achieving rows
    instead of relying on the base density's (possibly tiny) natural
    success rate.
    """
    if kind == "sudoku":
        expected_d = 33 if reward else 32
        if d not in (None, expected_d):
            raise ValueError(f"kind='sudoku'{' with reward=' + reward if reward else ''} is "
                             f"fixed at d={expected_d}, got d={d} (pass --d {expected_d}, or omit -d)")
        from . import sudoku as _sudoku

        row_base = _sudoku.SudokuRowsTarget(row_noise=row_noise)
        base = _sudoku.SudokuMixtureTarget(row_base, p_inject) if p_inject else row_base

        if reward:
            if reward != "sudoku_valid":
                raise ValueError("kind='sudoku' only supports reward='sudoku_valid' "
                                 "(the SudokuRewardAugmented construction)")
            return _sudoku.SudokuRewardAugmented(base)

        return base

    if kind == "product":
        base = ProductTarget.random(d, scale=1.0, seed=seed)
    elif kind == "ising":
        base = IsingTarget.random(d, J0=J0, field=field, seed=seed)
    else:
        raise ValueError(f"unknown base kind {kind!r}")
    return RewardAugmented(base, reward) if reward else base
