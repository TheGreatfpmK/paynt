"""Builder for colorings given by formulas: the authoring side of payntbind.synthesis.ColoringGeneral.

The standard Coloring gives every choice a conjunction of (parameter = option) pairs. A ColoringGeneral (SMPMC paper,
arXiv:2511.08078, Definition 2) instead colors every choice with an arbitrary quantifier-free formula over the finite
parameter space: the choice is enabled under a full parameter assignment iff the assignment satisfies its formula.
ColoringBuilder is how such colorings are defined, entirely in Python -- the C++ engine is generic and never needs
touching to support a new kind of coloring.

Colors are written as ordinary z3py formulas over the coloring's own parameter variables (one per declared parameter,
value = option index) and optional per-state/per-choice data columns, and ColoringBuilder translates them into the flat
node table ColoringGeneral's constructor expects. No z3 term ever crosses into C++ (payntbind links a different Z3
build than z3py, so the two cannot share objects): the walker below reads a z3py AST and re-emits it as plain data.

Supported formula language (anything else raises ValueError with the offending sub-expression):
    formulas: True, False, Not, And, Or (any arity), Implies, If (boolean result), ==, !=, <, <=, >, >=
    terms:    integer literals, parameter variables, data columns, +, -, unary -, *, If (integer result)
    atoms:    Or([p == o1, p == o2, ...]) for a single bare parameter p is recognised and compiled to an exact
              "p in {o1,o2,...}" atom instead of a chain of (possibly imprecise, see ColoringGeneral) equalities;
              ColoringBuilder.in_bits(p, data) similarly compiles to an exact "p's value is a set bit of the
              (per-state/choice) data word" atom (data is interpreted as a 64-bit bitmask, one bit per option).

Example: a custom "the environment parameter is in {0,2} or state_flag is set" coloring for choices [3, 7], over an
existing parameter_space:

    builder = ColoringBuilder(colored_mdp.underlying_mdp.nondeterministic_choice_indices, parameter_space.num_parameters)
    p = builder.parameters[0]
    flag = builder.state_column([...])  # one entry per state
    builder.color([3, 7], builder.template(z3.Or(z3.Or(p == 0, p == 2), flag == 1)))
    # every other choice is left uncolored (always enabled), matching Coloring's own convention
    coloring = builder.build()
    colored_mdp = paynt.colored_mdp.ColoredMdp(colored_mdp.underlying_mdp, parameter_space, coloring)

A colored MDP built from the result works with the OneByOne, CEGIS and SMPMC engines; generic AR and Hybrid read (parameter,
option) pairs back from the coloring and are rejected (see ColoredMdp.has_general_coloring).

The builder itself never holds a ParameterSpace (only a parameter *count*): a coloring built from a structural
template (e.g. a decision tree) determines its own parameter layout from the template, so its ParameterSpace can
only be reconstructed by the caller afterwards -- see paynt/dt/coloring_general.py and DtColoredMdpFactory.reset_tree.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import z3

import payntbind.synthesis
import paynt.parameter_space.parameter_space

import logging

logger = logging.getLogger(__name__)

NodeOp = payntbind.synthesis.ColoringGeneralNodeOp

# a marker function used purely as syntax: never handed to a solver, only ever pattern-matched by the walker below
_IN_BITS_FUNC = z3.Function("__paynt_coloring_builder_in_bits__", z3.IntSort(), z3.IntSort(), z3.BoolSort())


class _NodeTable:
    """Hash-consed flat node table (ColoringGeneral's constructor format), shared by every template of one builder."""

    def __init__(self) -> None:
        self.op: list[Any] = []
        self.a: list[int] = []
        self.b: list[int] = []
        self.c: list[int] = []
        self.option_sets: list[list[int]] = []
        self._node_cache: dict[tuple[Any, int, int, int], int] = {}
        self._option_set_cache: dict[tuple[int, ...], int] = {}

    def _add(self, op: Any, a: int, b: int, c: int) -> int:
        key = (op, a, b, c)
        node = self._node_cache.get(key)
        if node is not None:
            return node
        node = len(self.op)
        self.op.append(op)
        self.a.append(a)
        self.b.append(b)
        self.c.append(c)
        self._node_cache[key] = node
        return node

    def option_set(self, options: Iterable[int]) -> int:
        key = tuple(sorted({int(option) for option in options}))
        index = self._option_set_cache.get(key)
        if index is not None:
            return index
        index = len(self.option_sets)
        self.option_sets.append(list(key))
        self._option_set_cache[key] = index
        return index

    def const(self, value: int) -> int:
        return self._add(NodeOp.ConstTerm, int(value), 0, 0)

    def param(self, index: int) -> int:
        return self._add(NodeOp.ParamTerm, int(index), 0, 0)

    def state_col(self, column: int) -> int:
        return self._add(NodeOp.StateColTerm, int(column), 0, 0)

    def choice_col(self, column: int) -> int:
        return self._add(NodeOp.ChoiceColTerm, int(column), 0, 0)

    def add(self, a: int, b: int) -> int:
        return self._add(NodeOp.AddTerm, a, b, 0)

    def sub(self, a: int, b: int) -> int:
        return self._add(NodeOp.SubTerm, a, b, 0)

    def mul(self, a: int, b: int) -> int:
        return self._add(NodeOp.MulTerm, a, b, 0)

    def ite_term(self, cond: int, then: int, els: int) -> int:
        return self._add(NodeOp.IteTerm, cond, then, els)

    def true(self) -> int:
        return self._add(NodeOp.TrueF, 0, 0, 0)

    def false(self) -> int:
        return self._add(NodeOp.FalseF, 0, 0, 0)

    def not_(self, a: int) -> int:
        return self._add(NodeOp.NotF, a, 0, 0)

    def and_(self, a: int, b: int) -> int:
        return self._add(NodeOp.AndF, a, b, 0)

    def or_(self, a: int, b: int) -> int:
        return self._add(NodeOp.OrF, a, b, 0)

    def implies(self, a: int, b: int) -> int:
        return self._add(NodeOp.ImpliesF, a, b, 0)

    def ite_formula(self, cond: int, then: int, els: int) -> int:
        return self._add(NodeOp.IteF, cond, then, els)

    def eq(self, a: int, b: int) -> int:
        return self._add(NodeOp.EqF, a, b, 0)

    def ne(self, a: int, b: int) -> int:
        return self._add(NodeOp.NeF, a, b, 0)

    def lt(self, a: int, b: int) -> int:
        return self._add(NodeOp.LtF, a, b, 0)

    def le(self, a: int, b: int) -> int:
        return self._add(NodeOp.LeF, a, b, 0)

    def gt(self, a: int, b: int) -> int:
        return self._add(NodeOp.GtF, a, b, 0)

    def ge(self, a: int, b: int) -> int:
        return self._add(NodeOp.GeF, a, b, 0)

    def in_set(self, param_node: int, options: Iterable[int]) -> int:
        return self._add(NodeOp.InSetF, param_node, self.option_set(options), 0)

    def in_bits(self, param_node: int, data_node: int) -> int:
        return self._add(NodeOp.InBitsF, param_node, data_node, 0)


class Template:
    """A compiled color, ready to be attached to one or more choices via ColoringBuilder.color()."""

    def __init__(self, node: int) -> None:
        self.node = node


class _Walker:
    """Translates one z3py formula into ColoringBuilder's shared node table."""

    def __init__(self, builder: ColoringBuilder) -> None:
        self.builder = builder
        self.table = builder._table

    def _leaf(self, e: Any) -> int | None:
        name = str(e)
        if name in self.builder._param_index_by_name:
            return self.table.param(self.builder._param_index_by_name[name])
        if name in self.builder._state_col_by_name:
            return self.table.state_col(self.builder._state_col_by_name[name])
        if name in self.builder._choice_col_by_name:
            return self.table.choice_col(self.builder._choice_col_by_name[name])
        return None

    def _fold(self, children: list[Any], op: Any, walk: Any) -> int:
        node = walk(children[0])
        for child in children[1:]:
            node = op(node, walk(child))
        return node

    def _in_set_atom(self, e: Any) -> tuple[int, int] | None:
        """If e is (bare parameter) == (integer literal), in either order, return (its ParamTerm node id, the literal); else None -- used to recognise Or([p ==
        o, ...]) as an exact "p in {...}" atom."""
        if not z3.is_app(e) or e.decl().kind() != z3.Z3_OP_EQ:
            return None
        x, y = e.children()
        for lhs, rhs in ((x, y), (y, x)):
            leaf = self._leaf(lhs)
            if leaf is not None and self.table.op[leaf] == NodeOp.ParamTerm and z3.is_int_value(rhs):
                return leaf, rhs.as_long()
        return None

    def walk_term(self, e: Any) -> int:
        if z3.is_int_value(e):
            return self.table.const(e.as_long())
        leaf = self._leaf(e)
        if leaf is not None:
            return leaf
        if z3.is_app(e):
            kind = e.decl().kind()
            children = e.children()
            if kind == z3.Z3_OP_ADD:
                return self._fold(children, self.table.add, self.walk_term)
            if kind == z3.Z3_OP_SUB:
                if len(children) == 1:
                    return self.table.sub(self.table.const(0), self.walk_term(children[0]))
                return self._fold(children, self.table.sub, self.walk_term)
            if kind == z3.Z3_OP_UMINUS:
                return self.table.sub(self.table.const(0), self.walk_term(children[0]))
            if kind == z3.Z3_OP_MUL:
                return self._fold(children, self.table.mul, self.walk_term)
            if kind == z3.Z3_OP_ITE:
                cond, then, els = children
                return self.table.ite_term(self.walk_formula(cond), self.walk_term(then), self.walk_term(els))
        raise ValueError(f"unsupported term in a general coloring template: {e}")

    def walk_formula(self, e: Any) -> int:
        if z3.is_true(e):
            return self.table.true()
        if z3.is_false(e):
            return self.table.false()
        if not z3.is_app(e):
            raise ValueError(f"unsupported formula in a general coloring template: {e}")
        if e.decl() == _IN_BITS_FUNC:
            p_arg, d_arg = e.children()
            p_node = self.walk_term(p_arg)
            if self.table.op[p_node] != NodeOp.ParamTerm:
                raise ValueError(f"in_bits' first argument must be a bare parameter reference, got: {p_arg}")
            return self.table.in_bits(p_node, self.walk_term(d_arg))
        kind = e.decl().kind()
        children = e.children()
        if kind == z3.Z3_OP_NOT:
            return self.table.not_(self.walk_formula(children[0]))
        if kind == z3.Z3_OP_AND:
            return self._fold(children, self.table.and_, self.walk_formula)
        if kind == z3.Z3_OP_OR:
            in_set = self._recognise_in_set(children)
            if in_set is not None:
                return in_set
            return self._fold(children, self.table.or_, self.walk_formula)
        if kind == z3.Z3_OP_IMPLIES:
            a, b = children
            return self.table.implies(self.walk_formula(a), self.walk_formula(b))
        if kind == z3.Z3_OP_ITE:
            cond, then, els = children
            return self.table.ite_formula(self.walk_formula(cond), self.walk_formula(then), self.walk_formula(els))
        if kind == z3.Z3_OP_EQ:
            a, b = children
            return self.table.eq(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_DISTINCT:
            if len(children) != 2:
                raise ValueError("Distinct is only supported with exactly 2 arguments (use != instead)")
            a, b = children
            return self.table.ne(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_LT:
            a, b = children
            return self.table.lt(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_LE:
            a, b = children
            return self.table.le(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_GT:
            a, b = children
            return self.table.gt(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_GE:
            a, b = children
            return self.table.ge(self.walk_term(a), self.walk_term(b))
        raise ValueError(f"unsupported formula in a general coloring template: {e}")

    def _recognise_in_set(self, children: list[Any]) -> int | None:
        param_node: int | None = None
        options: list[int] = []
        for child in children:
            atom = self._in_set_atom(child)
            if atom is None:
                return None
            node, value = atom
            if param_node is None:
                param_node = node
            elif node != param_node:
                return None
            options.append(value)
        assert param_node is not None
        return self.table.in_set(param_node, options)


class ColoringBuilder:
    """Constructs a payntbind.synthesis.ColoringGeneral from z3py formulas -- see the module docstring."""

    def __init__(
        self,
        row_groups: list[int],
        num_parameters: int,
        state_is_relevant: Any = None,
    ) -> None:
        """
        :param row_groups: nondeterministic choice indices of the underlying quotient MDP (as on ColoredMdp.underlying_mdp)
        :param num_parameters: number of parameters colors are formulas over -- deliberately just a count, not a
            full ParameterSpace: a coloring built from a template (e.g. a decision tree, see paynt/dt/coloring_general.py)
            determines its own parameter *layout* (names, option counts) only from the coloring it just built, so a
            ParameterSpace is reconstructed by the caller only afterwards, never available to the builder itself
        :param state_is_relevant: optional stormpy/payntbind BitVector -- irrelevant states are never evaluated
            (see ColoringGeneral); every state is relevant by default
        """
        self.row_groups = list(row_groups)
        self.num_states = len(self.row_groups) - 1
        self.num_choices = self.row_groups[-1]
        self.num_parameters = num_parameters
        self._state_is_relevant = state_is_relevant
        self._table = _NodeTable()
        self._param_index_by_name: dict[str, int] = {}
        self._state_col_by_name: dict[str, int] = {}
        self._choice_col_by_name: dict[str, int] = {}
        self._state_rows: list[list[int]] = [[] for _ in range(self.num_states)]
        self._choice_rows: list[list[int]] = [[] for _ in range(self.num_choices)]
        self._choice_root: list[int | None] = [None] * self.num_choices
        self._parameters: list[Any] | None = None

    @property
    def parameters(self) -> list[Any]:
        """One z3py Int per declared parameter, value = option index."""
        if self._parameters is None:
            variables = []
            for index in range(self.num_parameters):
                name = f"__param_{index}__"
                self._param_index_by_name[name] = index
                variables.append(z3.Int(name))
            self._parameters = variables
        return self._parameters

    def state_column(self, values: list[int]) -> Any:
        """A per-state data column (one integer per state, in state order), usable as a z3py Int term."""
        if len(values) != self.num_states:
            raise ValueError(f"state_column expects {self.num_states} values, got {len(values)}")
        column = len(self._state_col_by_name)
        name = f"__state_col_{column}__"
        self._state_col_by_name[name] = column
        for state in range(self.num_states):
            self._state_rows[state].append(int(values[state]))
        return z3.Int(name)

    def choice_column(self, values: list[int]) -> Any:
        """A per-choice data column (one integer per choice, in choice order), usable as a z3py Int term."""
        if len(values) != self.num_choices:
            raise ValueError(f"choice_column expects {self.num_choices} values, got {len(values)}")
        column = len(self._choice_col_by_name)
        name = f"__choice_col_{column}__"
        self._choice_col_by_name[name] = column
        for choice in range(self.num_choices):
            self._choice_rows[choice].append(int(values[choice]))
        return z3.Int(name)

    @staticmethod
    def in_bits(parameter: Any, data: Any) -> Any:
        """True iff parameter's value is a set bit of data, interpreted as a 64-bit bitmask (one bit per option) -- e.g. a state's or choice's "unavailable
        options" mask.

        parameter must be a bare parameter variable.
        """
        return _IN_BITS_FUNC(parameter, data)

    def template(self, formula: Any) -> Template:
        """Compile a z3py formula (built over self.parameters / data columns / in_bits) into a reusable Template."""
        return Template(_Walker(self).walk_formula(formula))

    def color(self, choices: int | Iterable[int], template: Template) -> None:
        """Color one choice, or every choice in an iterable of choice indices, with template.

        Every choice may be colored at most once; choices never colored stay uncolored (always enabled), matching Coloring's own convention.
        """
        indices = [choices] if isinstance(choices, int) else list(choices)
        for choice in indices:
            if self._choice_root[choice] is not None:
                raise ValueError(f"choice {choice} is already colored")
            self._choice_root[choice] = template.node

    def build(self) -> Any:
        """Produce the payntbind.synthesis.ColoringGeneral."""
        import stormpy.storage

        choice_root = [-1 if root is None else root for root in self._choice_root]
        state_is_relevant = self._state_is_relevant
        if state_is_relevant is None:
            state_is_relevant = stormpy.storage.BitVector(0)
        return payntbind.synthesis.ColoringGeneral(
            self.row_groups,
            self.num_parameters,
            self._table.op,
            self._table.a,
            self._table.b,
            self._table.c,
            self._table.option_sets,
            choice_root,
            self._state_rows,
            self._choice_rows,
            state_is_relevant,
        )

    def check_definition(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, samples: int | None = None, seed: int = 0) -> None:
        """Check Definition 2 on sampled (or, when samples is None, all) full assignments of parameter_space: every state must have exactly one enabled choice.
        Raises AssertionError on the first violation found. Only practical on small parameter spaces (all_combinations() is exhaustive) or with a modest sample
        size.

        :param parameter_space: built by the caller to match this builder's layout (e.g. reset_tree's own reconstruction from a coloring's parameter_info, for a
            decision-tree coloring) -- the builder itself never holds one, see __init__.
        """
        import random

        coloring = self.build()
        combinations = list(parameter_space.all_combinations())
        if samples is not None and samples < len(combinations):
            combinations = random.Random(seed).sample(combinations, samples)
        for combination in combinations:
            assignment = parameter_space.construct_assignment(combination)
            selected = coloring.selectCompatibleChoices(assignment.native)
            for state in range(self.num_states):
                enabled = [choice for choice in range(self.row_groups[state], self.row_groups[state + 1]) if selected[choice]]
                assert len(enabled) == 1, (
                    f"Definition 2 violated at assignment {combination}, state {state}: {len(enabled)} choices enabled ({enabled}), expected exactly 1"
                )
