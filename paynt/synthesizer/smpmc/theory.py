"""The Z3 theory-solver plugin: a z3.UserPropagateBase subclass that turns Z3's own CDCL search into
the CDCL(T) integration described in Section 4.2-4.3 of arXiv:2511.08078.

Ported from molehill's plugins/search.py:SearchMarkovChain (https://github.com/linusheck/molehill,
GPL-3.0), with one correctness fix found empirically during this port and worth calling out explicitly:

Z3's preprocessor eagerly constant-folds a `viable(...)` argument into a numeral whenever that argument
is pinned by a simple equality -- not only for a literal constant written in the source formula, but for
*any* variable Z3 can already prove has exactly one possible value. In particular, every single-option
parameter (a hole whose in_range clause collapses to one equality) triggers this on the very first
`created` callback, for every query. An argument handled this way never appears in a `fixed` callback at
all, since it was never a live variable to begin with -- so a naive implementation that only tracks
*variable* arguments (as molehill's own bare-name-keyed dict implicitly does, and as an early draft of
this module did) silently drops that parameter from every `fixed` dict it builds. Silently, because the
solver still finds a model -- just one the theory was never actually asked about, since the theory's own
completeness argument depends on `fixed` reflecting every relevant parameter, constant-folded or not.
The fix is to track each `viable(...)` argument by its *position* (which is also its parameter index,
since it is always called as viable(*variables_in_parameter_index_order), see synthesizer.py) and record
constant-folded arguments directly, rather than only arguments that later arrive via `fixed`.
"""

from __future__ import annotations

from typing import Any

import z3

import paynt.synthesizer.smpmc.checker

import logging

logger = logging.getLogger(__name__)


class SmpmcPropagator(z3.UserPropagateBase):  # type: ignore[misc]
    def __init__(
        self,
        solver: Any,
        ctx: Any,
        theory: paynt.synthesizer.smpmc.checker.ColoredMdpTheory,
        name_to_parameter: dict[str, int],
    ):
        super().__init__(solver, ctx)
        self.add_fixed(self._fixed)
        self.add_final(self._final)
        self.add_created(self._created)

        self.theory = theory
        self.name_to_parameter = name_to_parameter

        # ast key -> current value (bool for a viable(...) literal, int for a parameter variable)
        self.partial_model: dict[str, Any] = {}
        # undo log: keys pushed onto partial_model, in order, so pop() can unwind exactly num_scopes levels
        self.trail: list[str] = []
        # trail marks, one per push()
        self.scopes: list[int] = []
        # ast key of a viable(...) application -> its argument specs, positional (position == parameter
        # index): ("const", value) for a Z3-constant-folded argument, ("var", arg_key) for a live one
        self.viable_args: dict[str, list[tuple[str, Any]]] = {}
        # ast key -> the actual z3 term, needed to build conflict() dependencies
        self.term_of: dict[str, Any] = {}

    @staticmethod
    def _key(ast: Any) -> str:
        """A cheap, stable key for a Z3 AST node -- the Z3 APIs for structural identity are too slow to
        call on this hot path (molehill's own comment on the equivalent code), so this uses the printed
        s-expression for an application (distinguishing e.g. different viable(...) argument tuples) and
        the declaration name for a plain variable."""
        text = ast.sexpr()
        return text if text.startswith("(") else ast.decl().name()

    def _created(self, term: Any) -> None:
        key = self._key(term)
        args: list[tuple[str, Any]] = []
        for i in range(term.num_args()):
            arg = term.arg(i)
            if z3.Z3_is_numeral_ast(arg.ctx_ref(), arg.as_ast()):
                args.append(("const", arg.as_long()))
            else:
                arg_key = self._key(arg)
                self.term_of[arg_key] = arg
                self.add(arg)
                args.append(("var", arg_key))
        self.viable_args[key] = args
        self.term_of[key] = term

    def _fixed(self, ast: Any, value: Any) -> None:
        key = self._key(ast)
        self.term_of[key] = ast
        if z3.is_true(value):
            decoded: Any = True
        elif z3.is_false(value):
            decoded = False
        else:
            decoded = value.as_long()
        self.partial_model[key] = decoded
        self.trail.append(key)

    def push(self) -> None:
        self.scopes.append(len(self.trail))
        self._analyse()

    def pop(self, num_scopes: int) -> None:
        for _ in range(num_scopes):
            mark = self.scopes.pop()
            while len(self.trail) > mark:
                key = self.trail.pop()
                del self.partial_model[key]

    def _final(self) -> None:
        self._analyse()

    def _analyse(self) -> None:
        """Check every currently-fixed viable(...) literal against the theory; on the first refutation,
        push a learned conflict clause back into Z3 and stop (Z3 will call back in for more once it has
        backtracked past this conflict).

        Deliberately does NOT wait for every argument of viable(...) to be fixed before consulting the
        theory -- Algorithm 1 in the paper is explicit that it works over whatever partial model K
        currently holds ("for all literals k in K do ... eta <- {theta | theta satisfies K \\ {v}}"), not
        just a complete one. An earlier version of this method required every parameter to be fixed first,
        which is a correctness-preserving but severely performance-destroying simplification: since
        viable(...) is asserted over every parameter, "wait for all of them" means the theory is *never*
        consulted until Z3 has already committed to one fully concrete candidate, degenerating the whole
        search into one-candidate-at-a-time enumeration with no early pruning from partial information --
        exactly the CEGIS-like behavior SMPMC is meant to avoid, confirmed empirically (see the
        conversation this was found in: a controlled comparison on a small synthetic instance showed a
        4x reduction in total theory calls -- most of them genuinely partial -- once this requirement was
        removed, with identical sat/unsat answers either way). Parameters not yet fixed are simply left
        out of `fixed`; checker.py's eta construction already treats an unmentioned parameter as "still
        fully open" and reasons about the induced sub-MDP accordingly.

        One exception, ported from molehill's own Mole.partial_model_consistent (its "magic trick" that
        checks a DTMC first): a *partial* (non-empty, not-fully-fixed) query is skipped entirely --
        reported inconclusive without ever calling the theory -- until at least one fully-fixed assignment
        has been checked. This isn't in tension with the paragraph above; it's a different failure mode
        the partial-consultation fix doesn't address on its own. Partial consultation pays off when a
        partial query can be *refuted*, pruning a whole region for free -- but on a property that's easy
        to satisfy (found on a real 1.65M-state DTMC-with-holes model, arXiv:2511.08078's own benchmark
        suite: a reward threshold so generous that virtually every completion satisfies it), no partial
        query is ever refutable, so every one of them is pure, expensive overhead: an 8-second VI call on
        a barely-smaller sub-MDP, paid once per parameter Z3 fixes on the way to its first candidate, for
        no pruning benefit at all. Deferring partial checks until Z3 has found *some* concrete candidate
        first sidesteps this without giving up early pruning altogether -- it only delays it until pruning
        has a chance to actually pay for itself.
        """
        for key, value in list(self.partial_model.items()):
            if not isinstance(value, bool):
                continue
            arg_specs = self.viable_args.get(key)
            if arg_specs is None:
                continue

            fixed: dict[int, int] = {}
            for parameter, (kind, payload) in enumerate(arg_specs):
                if kind == "const":
                    fixed[parameter] = payload
                elif payload in self.partial_model:
                    fixed[parameter] = self.partial_model[payload]
                # else: this parameter isn't fixed yet -- leave it out of `fixed`, don't abort the analysis

            is_full_assignment = len(fixed) == len(arg_specs)
            if not is_full_assignment and fixed and not self.theory.first_full_assignment_checked:
                continue

            result = self.theory.check(fixed, value)
            if is_full_assignment:
                self.theory.first_full_assignment_checked = True
            if result is None:
                continue

            deps = [self.term_of[key]]
            for parameter in result.conflict_parameters:
                kind, payload = arg_specs[parameter]
                if kind == "var":
                    deps.append(self.term_of[payload])
            self.conflict(deps=deps)
            return

    def fresh(self, new_ctx: Any) -> SmpmcPropagator:
        """Called by Z3 to create a sub-context propagator, notably during MBQI quantifier instantiation
        for robust synthesis (Section 4.4). Shares theory (hence its cache) with the parent -- MBQI's
        sub-contexts explore the same colored MDP, so there is no reason to lose cached results, and the
        theory's own epoch/cache state must stay consistent across every propagator instance in a run."""
        return SmpmcPropagator(None, new_ctx, self.theory, self.name_to_parameter)
