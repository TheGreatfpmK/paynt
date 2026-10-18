#!/usr/bin/env python3
"""Memoryless stochastic policies for interval POMDPs: alternating best responses of the agent and nature.

The robust value of a memoryless stochastic policy sigma is r(sigma) = min_theta V(sigma, theta). Nature
picks every interval row (s,a) independently, so for a fixed sigma the induced model is an interval DTMC and
nature's best response sits at the interval bounds (robust policy iteration). For a fixed theta, finding
the best sigma is a parametric Markov chain problem; here it is solved locally by L-BFGS-B with exact
(adjoint) gradients. Every action keeps probability >= delta, so the graph of the induced chain does not
depend on sigma and the value is smooth.

    alternate      sigma_{k+1} = best response to theta_k = BR(sigma_k) (the loop being studied)
    double-oracle  sigma_{k+1} = best response to all of theta_0..theta_k (max of the min over the set);
                   stops when nature's best response is already in the set: then sigma is a local maximum
                   of the robust value (a global one if the agent's step is solved globally)
    mdp            upper bound: the robust value of the fully observable interval MDP

Reachability only: the agent maximizes P[left U right], nature minimizes it.

Examples:
    python robust-memoryless.py models/robust-pomdps/robust-pennies.prism 'P=? [F "won"]'
    python robust-memoryless.py models/robust-pomdps/evade-large.prism 'P=? ["notbad" U "goal"]' --set N=4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import time

import numpy as np
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_spec = importlib.util.spec_from_file_location("interval_ties", os.path.join(os.path.dirname(os.path.abspath(__file__)), "interval-ties.py"))
ties = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ties)

logger = logging.getLogger("robust-memoryless")


class Problem:
    """An interval POMDP with (s,a)-rectangular nature and a reachability objective."""

    def __init__(self, model, params, objective, delta: float):
        self.delta = delta
        matrix = model.transition_matrix
        n, C = model.nr_states, matrix.nr_rows
        self.n, self.C = n, C
        self.init = model.initial_states[0]
        group_sizes = [matrix.get_row_group_end(s) - matrix.get_row_group_start(s) for s in range(n)]
        self.choice_state = np.repeat(np.arange(n), group_sizes)
        self.group_start = np.concatenate([[0], np.cumsum(group_sizes)[:-1]])

        # policy variables: one per (observation, action label)
        labels = ties.labels_at_observation(model)
        offsets = np.concatenate([[0], np.cumsum([len(l) for l in labels])])
        self.nv = int(offsets[-1])
        self.var_obs = np.repeat(np.arange(len(labels)), [len(l) for l in labels])
        self.var_of_choice = np.array([offsets[model.observations[s]] + labels[model.observations[s]].index(ties.choice_label(model, c)) for c, s in enumerate(self.choice_state)])
        self.labels = labels
        self.offsets = offsets

        # entries: probability alpha + beta * theta_c of choice c; nature has one theta per parametric choice
        rows = ties.affine_entries(model, params)
        e_choice, e_col, e_alpha, e_beta = [], [], [], []
        self.lo, self.hi = np.zeros(C), np.zeros(C)
        self.parametric = np.zeros(C, dtype=bool)
        for c, entries in enumerate(rows):
            for col, alpha, beta, pidx in entries:
                e_choice.append(c)
                e_col.append(col)
                e_alpha.append(alpha)
                e_beta.append(beta)
                if beta != 0:
                    self.parametric[c] = True
                    self.lo[c], self.hi[c] = params[pidx].lo, params[pidx].hi
        self.e_choice, self.e_col = np.array(e_choice), np.array(e_col)
        self.e_alpha, self.e_beta = np.array(e_alpha), np.array(e_beta)
        assert np.all(self.lo[self.parametric] > 0) and np.all(self.hi[self.parametric] < 1)

        # qualitative analysis with every action possible (every action has probability >= delta)
        self.target = objective.right.copy()
        stop = objective.right | ~objective.left
        graph = sp.csr_matrix((np.ones(len(e_col)), (self.choice_state[self.e_choice], self.e_col)), shape=(n, n))
        can_reach = ties.Chain.backward_reachable(graph, self.target, ~stop | self.target)
        self.maybe = can_reach & ~stop
        self.m_index = np.full(n, -1)
        self.maybe_states = np.flatnonzero(self.maybe)
        self.m_index[self.maybe_states] = np.arange(len(self.maybe_states))
        self.m = len(self.maybe_states)
        from_maybe = self.maybe[self.choice_state[self.e_choice]]
        self.A_mask = from_maybe & self.maybe[self.e_col]
        self.b_mask = from_maybe & self.target[self.e_col]
        self.A_rows = self.m_index[self.choice_state[self.e_choice[self.A_mask]]]
        self.A_cols = self.m_index[self.e_col[self.A_mask]]
        self.b_rows = self.m_index[self.choice_state[self.e_choice[self.b_mask]]]
        self.choice_in_maybe = self.maybe[self.choice_state]
        self.solves = 0


    # ----- policies and nature
    def uniform(self):
        sizes = np.diff(self.offsets)
        return 1.0 / sizes[self.var_obs]

    def random_policy(self, rng):
        sigma = rng.random(self.nv) + 1e-3
        sums = np.bincount(self.var_obs, weights=sigma)
        return self.floor(sigma / sums[self.var_obs])

    def floor(self, sigma):
        """Project onto {sigma >= delta, sums 1} per observation (only shifts mass, keeps the support full)."""
        sizes = np.diff(self.offsets)[self.var_obs]
        delta = np.minimum(self.delta, 1.0 / sizes)
        sigma = np.maximum(sigma, delta)
        excess = np.bincount(self.var_obs, weights=sigma) - 1.0
        room = sigma - delta
        room_sum = np.bincount(self.var_obs, weights=room)
        share = np.where(room_sum[self.var_obs] > 0, room / np.where(room_sum[self.var_obs] > 0, room_sum[self.var_obs], 1), 0)
        return sigma - excess[self.var_obs] * share

    # ----- evaluation
    def solve(self, sigma, theta):
        """Values of all states, the factorization, and the entry probabilities."""
        self.solves += 1
        probs = self.e_alpha + self.e_beta * theta[self.e_choice]
        weighted = sigma[self.var_of_choice[self.e_choice]] * probs
        x = np.zeros(self.n)
        x[self.target] = 1.0
        if self.m == 0:
            return x, None, probs
        A = sp.identity(self.m, format="csc") - sp.csc_matrix((weighted[self.A_mask], (self.A_rows, self.A_cols)), shape=(self.m, self.m))
        b = np.bincount(self.b_rows, weights=weighted[self.b_mask], minlength=self.m)
        lu = spla.splu(A.tocsc())
        x[self.maybe_states] = lu.solve(b)
        return x, lu, probs

    def value(self, sigma, theta):
        return self.solve(sigma, theta)[0][self.init]

    def value_grad(self, sigma, theta):
        """V(sigma, theta) and its gradient in sigma (expected visits times the Q-values of the choices)."""
        x, lu, probs = self.solve(sigma, theta)
        if lu is None or not self.maybe[self.init]:
            return x[self.init], np.zeros(self.nv)
        e = np.zeros(self.m)
        e[self.m_index[self.init]] = 1.0
        y = np.zeros(self.n)
        y[self.maybe_states] = lu.solve(e, trans="T")
        q = np.bincount(self.e_choice, weights=probs * x[self.e_col], minlength=self.C)
        weight = np.where(self.choice_in_maybe, y[self.choice_state] * q, 0.0)
        return x[self.init], np.bincount(self.var_of_choice, weights=weight, minlength=self.nv)

    def nature(self, sigma, theta=None):
        """Nature's best response to sigma (robust policy iteration over the interval bounds)."""
        theta = self.lo.copy() if theta is None else theta.copy()
        for _ in range(10000):
            x, _, _ = self.solve(sigma, theta)
            d = np.bincount(self.e_choice, weights=self.e_beta * x[self.e_col], minlength=self.C)
            scale = 1e-13 * max(np.abs(x).max(), 1e-300)
            want = np.where(d > scale, self.lo, np.where(d < -scale, self.hi, theta))
            change = self.parametric & self.choice_in_maybe & (np.abs(want - theta) * np.abs(d) > scale)
            if not change.any():
                return x[self.init], theta
            theta = np.where(change, want, theta)
        raise RuntimeError("nature's policy iteration did not converge")

    def nature_key(self, theta):
        """Nature's choices on the rows that matter, as hashable bits."""
        relevant = self.parametric & self.choice_in_maybe
        return np.packbits(theta[relevant] <= self.lo[relevant]).tobytes()

    # ----- the agent's step
    def sigma_of_weights(self, w):
        """sigma = delta + (1 - k delta) w / sum(w) per observation (k actions), with box bounds on w."""
        k = np.diff(self.offsets)[self.var_obs]
        d = np.minimum(self.delta, 1.0 / k)
        total = np.bincount(self.var_obs, weights=w)[self.var_obs]
        return d + (1 - k * d) * w / total, (1 - k * d), total

    def best_response(self, sigma, thetas, max_evaluations=20000):
        """Local maximum of min_j V(sigma, theta_j): L-BFGS-B on normalized weights; with several theta_j the
        minimum is smoothed by a soft-min whose temperature decreases to 1e-7 of the value."""
        evaluations = 0

        def objective(w, tau):
            nonlocal evaluations
            evaluations += 1
            sig, c, total = self.sigma_of_weights(w)
            values, grads = zip(*[self.value_grad(sig, th) for th in thetas])
            values = np.array(values)
            if len(thetas) == 1:
                f, g = values[0], grads[0]
            else:
                low = values.min()
                weights = np.exp(-(values - low) / tau)
                f = low - tau * np.log(weights.sum())
                g = sum(p * gr for p, gr in zip(weights / weights.sum(), grads))
            # chain rule through the normalization of each observation
            gs = g * c / total
            dot = np.bincount(self.var_obs, weights=g * c * w / total**2)[self.var_obs]
            return -f, -(gs - dot)

        w = sigma.copy()
        scale = max(abs(min(self.value(sigma, th) for th in thetas)), 1e-12)
        for tau in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] if len(thetas) > 1 else [None]:
            res = scipy.optimize.minimize(
                objective, w, args=(None if tau is None else tau * scale,), jac=True, method="L-BFGS-B",
                bounds=[(1e-12, 1.0)] * self.nv, options=dict(maxiter=max_evaluations, maxfun=max_evaluations, ftol=1e-15, gtol=1e-12),
            )
            w = res.x
        sigma = self.sigma_of_weights(w)[0]
        return sigma, min(self.value(sigma, th) for th in thetas), evaluations

    def best_response_restarts(self, sigma, thetas, restarts, rng):
        best = self.best_response(sigma, thetas)
        starts = [self.uniform()] + [self.random_policy(rng) for _ in range(restarts - 1)] if restarts > 0 else []
        for start in starts:
            candidate = self.best_response(self.floor(start), thetas)
            if candidate[1] > best[1] + 1e-12:
                best = candidate
        return best

    # ----- bounds
    def mdp_upper_bound(self, iterations=1000000, tol=1e-12):
        """Robust value of the fully observable interval MDP (value iteration from below)."""
        x = np.zeros(self.n)
        x[self.target] = 1.0
        for _ in range(iterations):
            d = np.bincount(self.e_choice, weights=self.e_beta * x[self.e_col], minlength=self.C)
            theta = np.where(d > 0, self.lo, self.hi)
            q = np.bincount(self.e_choice, weights=(self.e_alpha + self.e_beta * theta[self.e_choice]) * x[self.e_col], minlength=self.C)
            best = np.maximum.reduceat(q, self.group_start)
            new = np.where(self.maybe, best, x)
            if np.abs(new - x).max() < tol:
                return new[self.init]
            x = new
        return x[self.init]


def describe(problem: Problem, sigma, limit=6):
    """The randomizing observations of a policy."""
    parts = []
    for z, labels in enumerate(problem.labels):
        if len(labels) < 2:
            continue
        probs = sigma[problem.offsets[z] : problem.offsets[z + 1]]
        if probs.max() < 1 - 1e-3:
            parts.append(f"z{z}:" + ",".join(f"{l}={p:.3f}" for l, p in zip(labels, probs)))
    text = f"{len(parts)} randomizing observations"
    return text + (": " + " ".join(parts[:limit]) if parts else "")


def alternate(problem: Problem, sigma, args, rng):
    history = []
    seen = {}
    best = (-np.inf, None)
    for k in range(args.iterations):
        r, theta = problem.nature(sigma)
        key = problem.nature_key(theta)
        if r > best[0]:
            best = (r, sigma.copy())
        sigma_next, optimistic, evaluations = problem.best_response_restarts(sigma, [theta], args.restarts, rng)
        history.append({"k": k, "robust": r, "optimistic": optimistic, "evaluations": evaluations, "nature": seen.setdefault(key, len(seen))})
        logger.info(f"alternate k={k}: robust value {r:.6g}; best response to nature #{history[-1]['nature']} reaches {optimistic:.6g} on that model ({evaluations} evaluations)")
        if np.abs(sigma_next - sigma).max() < 1e-7:
            logger.info("alternate: fixed point (a saddle point of the local game)")
            break
        if k >= 2 and history[-1]["nature"] == history[-3]["nature"] and abs(history[-1]["robust"] - history[-3]["robust"]) < 1e-9 * max(1, abs(r)):
            logger.info("alternate: cycle of nature's responses detected")
            break
        sigma = sigma_next
    return best, history


def double_oracle(problem: Problem, sigma, args, rng):
    thetas, keys, history = [], set(), []
    best = (-np.inf, None)
    for k in range(args.iterations):
        r, theta = problem.nature(sigma)
        if r > best[0]:
            best = (r, sigma.copy())
        key = problem.nature_key(theta)
        if key in keys:
            history.append({"k": k, "robust": r, "set": len(thetas)})
            logger.info(f"double-oracle k={k}: robust value {r:.6g}; nature's response is already in the set -> local optimum")
            break
        keys.add(key)
        thetas.append(theta)
        sigma, upper, evaluations = problem.best_response_restarts(sigma, thetas, args.restarts, rng)
        history.append({"k": k, "robust": r, "set": len(thetas), "max_min_over_set": upper, "evaluations": evaluations})
        logger.info(f"double-oracle k={k}: robust value {r:.6g}; max-min over {len(thetas)} responses {upper:.6g} ({evaluations} evaluations)")
    return best, history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("prop", help='P=? [F a] or P=? ["label" U b]; the agent maximizes, nature minimizes')
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a constant of the model")
    parser.add_argument("--delta", type=float, default=1e-8, help="minimum probability of every action")
    parser.add_argument("--iterations", type=int, default=30, help="outer iterations of each scheme")
    parser.add_argument("--restarts", type=int, default=0, help="extra starts of the agent's step (uniform, then random)")
    parser.add_argument("--start", default="uniform", help="uniform or random:SEED")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="append a JSON line with the results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(args.model) as f:
        text, params = ties.parametrize_prism(ties.override_constants(f.read(), args.set), False)
    program, model = ties.build_parametric(text, params, args.prop)
    problem = Problem(model, params, ties.Objective(model, program, args.prop), args.delta)
    logger.info(f"{os.path.basename(args.model)} {args.set}: {model.nr_states} states, {model.nr_observations} observations, {problem.nv} policy variables, {int(problem.parametric.sum())} interval rows")
    upper = problem.mdp_upper_bound()
    logger.info(f"fully observable robust MDP (upper bound): {upper:.6g}")

    rng = np.random.default_rng(args.seed)
    start = problem.uniform() if args.start == "uniform" else problem.random_policy(np.random.default_rng(int(args.start.split(":")[1])))
    start = problem.floor(start)
    record = {"model": args.model, "set": args.set, "prop": args.prop, "start": args.start, "restarts": args.restarts, "mdp_upper_bound": upper}
    for name, scheme in (("alternate", alternate), ("double-oracle", double_oracle)):
        t0 = time.time()
        (best_value, best_sigma), history = scheme(problem, start.copy(), args, rng)
        record[name] = {"best_robust": best_value, "iterations": len(history), "history": history, "time": time.time() - t0}
        logger.info(f"{name}: best robust value {best_value:.6g} after {len(history)} iterations ({time.time() - t0:.1f}s); {describe(problem, best_sigma)}")
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
