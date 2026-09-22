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
        backtracked past this conflict)."""
        for key, value in list(self.partial_model.items()):
            if not isinstance(value, bool):
                continue
            arg_specs = self.viable_args.get(key)
            if arg_specs is None:
                continue

            fixed: dict[int, int] = {}
            ready = True
            for parameter, (kind, payload) in enumerate(arg_specs):
                if kind == "const":
                    fixed[parameter] = payload
                else:
                    if payload not in self.partial_model:
                        ready = False
                        break
                    fixed[parameter] = self.partial_model[payload]
            if not ready:
                continue

            result = self.theory.check(fixed, value)
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
