"""Unit tests for ColoringBuilder's z3py-to-node-table walker, exercising formula shapes fromColoring never produces (Or, Not, Implies, Ite, in_bits, arithmetic
comparisons) directly against the compiled coloring's selectCompatibleChoices/check_definition, on synthetic row_groups
(selectCompatibleChoices/getStateToHoles/ relevantParameters/areChoicesConsistent only need the coloring and row_groups, not a real transition matrix -- so a
real stormpy model isn't needed here).

End-to-end synthesis on such colorings is exercised via the DT template (see tests/dt), the first real client of these formula shapes.
"""

from __future__ import annotations

import itertools

import pytest
import z3

import paynt.utils.coloring_builder
import paynt.parameter_space.parameter_space


def make_parameter_space(domains):
    """A ParameterSpace whose domains are, deliberately, exactly [0..len(domain)-1] for every given domain -- so a parameter's *option index* (what
    holeOptions/parameter_set_options operate on) and the plain integer used in these tests' z3py formulas coincide, keeping the tests' arithmetic legible."""
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for index, domain in enumerate(domains):
        assert list(domain) == list(range(len(domain))), "test domains must be range(len(domain)) -- see docstring"
        parameter_space.add_parameter(f"p{index}", list(domain))
    return parameter_space


class TestOrRecognisesInSet:
    def test_or_of_equalities_is_exact_even_where_a_kleene_or_of_plain_equalities_would_be_unknown(self):
        # 1 state, 2 choices; parameter p in {0,1,2}; choice0 <- Or(p==0, p==1), choice1 <- p==2
        parameter_space = make_parameter_space([[0, 1, 2]])
        row_groups = [0, 2]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = builder.parameters[0]
        builder.color(0, builder.template(z3.Or(p == 0, p == 1)))
        builder.color(1, builder.template(p == 2))
        coloring = builder.build()

        # restrict p to exactly {0,1}: In_set(p,{0,1}) is exactly TRUE here (choice0 certain), and choice1 (p==2) is
        # excluded -- a naive Or(Eq,Eq) translation would instead leave both at Kleene-U (kept, imprecisely)
        subfamily = parameter_space.copy()
        subfamily.parameter_set_options(0, [0, 1])
        selection = coloring.selectCompatibleChoices(subfamily.native)
        assert [selection[choice] for choice in range(2)] == [True, False]


class TestGenericFormulaShapes:
    def test_not_implies_and_ite_match_a_hand_computed_truth_table(self):
        # 1 state, 4 choices, 2 parameters p,q in {0,1}, deliberately routed through Not/Implies/Ite/And so the
        # walker's coverage of those shapes is exercised; each color is logically equivalent to a simple condition
        parameter_space = make_parameter_space([[0, 1], [0, 1]])
        row_groups = [0, 4]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p, q = builder.parameters
        builder.color(0, builder.template(z3.Not(p == 0)))  # p == 1
        builder.color(1, builder.template(z3.Implies(p == 0, q == 1)))  # p==1 or q==1
        builder.color(2, builder.template(z3.If(p == 0, q == 0, q == 1)))  # (p==0 and q==0) or (p==1 and q==1)
        builder.color(3, builder.template(z3.And(p == 0, q == 0)))
        coloring = builder.build()

        for pv, qv in itertools.product((0, 1), (0, 1)):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [pv])
            assignment.parameter_set_options(1, [qv])
            raw = coloring.selectCompatibleChoices(assignment.native)
            selection = [raw[choice] for choice in range(4)]
            expected = [
                pv == 1,
                pv == 1 or qv == 1,
                (pv == 0 and qv == 0) or (pv == 1 and qv == 1),
                pv == 0 and qv == 0,
            ]
            assert selection == expected, (pv, qv, selection, expected)

    def test_comparisons_and_arithmetic_over_a_data_column(self):
        # 1 state, 2 choices; parameter p in {0,1,2,3}; choice0 <- p + 1 <= column (column == 2, so p in {0,1}),
        # choice1 <- its logical complement, so every assignment has exactly one enabled choice, as a well-formed
        # coloring must (a lone choice0 would leave p in {2,3} with none, tripping ColoringGeneral's "no choice is
        # available" invariant -- the same one ColoringSmt enforces, guaranteed there by the tree's don't-care leaf)
        parameter_space = make_parameter_space([[0, 1, 2, 3]])
        row_groups = [0, 2]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = builder.parameters[0]
        column = builder.state_column([2])
        condition = p + 1 <= column
        builder.color(0, builder.template(condition))
        builder.color(1, builder.template(z3.Not(condition)))
        coloring = builder.build()
        for pv in range(4):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [pv])
            selection = coloring.selectCompatibleChoices(assignment.native)
            assert selection[0] == (pv <= 1)
            assert selection[1] == (pv > 1)


class TestInBits:
    def test_in_bits_matches_a_hand_computed_bitmask(self):
        # 1 state, 2 choices; parameter p in {0,1,2,3}; the state's data column is a bitmask with bits 0 and 2 set
        # (0b0101 = 5); choice0 <- p's value is one of those set bits, choice1 <- its complement (see above)
        parameter_space = make_parameter_space([[0, 1, 2, 3]])
        row_groups = [0, 2]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = builder.parameters[0]
        mask = builder.state_column([5])
        condition = builder.in_bits(p, mask)
        builder.color(0, builder.template(condition))
        builder.color(1, builder.template(z3.Not(condition)))
        coloring = builder.build()
        for pv in range(4):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [pv])
            selection = coloring.selectCompatibleChoices(assignment.native)
            assert selection[0] == (pv in (0, 2))
            assert selection[1] == (pv not in (0, 2))


class TestCheckDefinition:
    def test_passes_for_a_well_formed_coloring_and_raises_for_a_broken_one(self):
        parameter_space = make_parameter_space([[0, 1]])
        row_groups = [0, 2]

        good = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = good.parameters[0]
        good.color(0, good.template(p == 0))
        good.color(1, good.template(p == 1))
        good.check_definition(parameter_space)

        broken = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = broken.parameters[0]
        broken.color(0, broken.template(p == 0))
        broken.color(1, broken.template(p == 0))  # both choices share the same color: 2 enabled at p=0, 0 at p=1
        with pytest.raises(AssertionError):
            broken.check_definition(parameter_space)


class TestColoringErrors:
    def test_coloring_the_same_choice_twice_raises(self):
        parameter_space = make_parameter_space([[0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1], parameter_space.num_parameters)
        p = builder.parameters[0]
        builder.color(0, builder.template(p == 0))
        with pytest.raises(ValueError):
            builder.color(0, builder.template(p == 1))

    def test_an_unsupported_formula_shape_raises_value_error(self):
        parameter_space = make_parameter_space([[0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1], parameter_space.num_parameters)
        p = builder.parameters[0]
        x = z3.Real("x")  # a sort this DSL doesn't support at all
        with pytest.raises(ValueError):
            builder.template(z3.And(p == 0, x > 0))
