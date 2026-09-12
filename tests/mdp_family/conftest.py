import pytest

import paynt.parser.sketch

from helpers.helper import get_sketch_paths


@pytest.fixture
def family_colored_mdp_factory():
    """A sketch whose PRISM program is an MDP (not a DTMC) with parameters resolves to a FamilyColoredMdp --
    the "family of MDPs" fragment, searching for a policy that works across every family member."""
    sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def family_task():
    """The SynthesisTask sibling of family_colored_mdp_factory -- see tests/dt/conftest.py's dt_task for why
    this loads the sketch itself rather than reaching into the factory (which now only carries the
    family-specific FamilyTask, not the base task fields like disable_expected_visits)."""
    sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
    _, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def family_colored_mdp(family_colored_mdp_factory):
    return family_colored_mdp_factory.colored_mdp


@pytest.fixture
def pomdp_family_colored_mdp_factory():
    """A sketch whose PRISM program is a POMDP with parameters resolves to a PomdpFamilyColoredMdp -- here the
    parameters select which of 64 environment variants is in play (observation-equivalence classes tie policy
    decisions together across variants), not a policy decision themselves; those instead come from an FSC
    applied uniformly across the family via build_dtmc_sketch (see test_pomdp_family.py)."""
    sketch_path, props_path = get_sketch_paths("tests/pomdp-family-avoid-smaller")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def pomdp_family_task():
    """The SynthesisTask sibling of pomdp_family_colored_mdp_factory."""
    sketch_path, props_path = get_sketch_paths("tests/pomdp-family-avoid-smaller")
    _, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def pomdp_family_colored_mdp(pomdp_family_colored_mdp_factory):
    return pomdp_family_colored_mdp_factory.colored_mdp
