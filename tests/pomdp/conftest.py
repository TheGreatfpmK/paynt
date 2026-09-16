import pytest

import paynt.parser.sketch

from helpers.helper import get_sketch_paths


@pytest.fixture
def pomdp_colored_mdp_factory():
    sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def pomdp_task():
    """The SynthesisTask sibling of pomdp_colored_mdp_factory -- see tests/dt/conftest.py's dt_task for why
    this loads the sketch itself rather than reaching into the factory."""
    sketch_path, props_path = get_sketch_paths("tests/pomdp-maze")
    _, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def pomdp_colored_mdp(pomdp_colored_mdp_factory):
    return pomdp_colored_mdp_factory.build()


@pytest.fixture
def decpomdp_colored_mdp_factory():
    """A genuine multi-agent Cassandra/.dpomdp sketch, which resolves through
    payntbind.synthesis.parse_decpomdp to a real decpomdp_manager (num_agents > 1) and so through
    DecPomdpColoredMdpFactory. This is NOT the same as models/archive/*/dec-pomdp/*, whose sketch.templ
    files are actually plain PRISM DTMC-with-parameters sketches (parse cleanly as PRISM, model_type DTMC) --
    similarly-named parameters, but they resolve to a plain paynt.colored_mdp.ColoredMdp (no specialist
    factory at all) and never touch this code at all."""
    # props_name deliberately omitted (defaults to "sketch.props", which does not exist in this directory)
    # so this exercises the Cassandra branch's model-inferred discounted-reward property (discount 0.9)
    sketch_path, props_path = get_sketch_paths("tests/decpomdp-dectiger", sketch_name="dectiger.dpomdp")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def decpomdp_task():
    """The SynthesisTask sibling of decpomdp_colored_mdp_factory."""
    sketch_path, props_path = get_sketch_paths("tests/decpomdp-dectiger", sketch_name="dectiger.dpomdp")
    _, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def decpomdp_colored_mdp(decpomdp_colored_mdp_factory):
    return decpomdp_colored_mdp_factory.build()


@pytest.fixture
def posmg_colored_mdp_factory():
    """A POSMG sketch resolves to a PosmgColoredMdp via PosmgColoredMdpFactory. mec-test has only
    single-state (perfect-information) observations for the optimizing player, so it is fast but never
    actually exercises memory unfolding -- see posmg_test_game_colored_mdp_factory/test_posmg_synthesizer.py
    for that."""
    sketch_path, props_path = get_sketch_paths("tests/posmg-mec-test")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def posmg_colored_mdp(posmg_colored_mdp_factory):
    return posmg_colored_mdp_factory.build()


@pytest.fixture
def posmg_test_game_colored_mdp_factory():
    """Unlike mec-test, test-game's optimizing player has multi-state observations, so this is the fixture
    that actually exercises memory unfolding (see test_synthesize_improves_with_memory_unfolding)."""
    sketch_path, props_path = get_sketch_paths("tests/posmg-test-game")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def posmg_test_game_task():
    """The SynthesisTask sibling of posmg_test_game_colored_mdp_factory."""
    sketch_path, props_path = get_sketch_paths("tests/posmg-test-game")
    _, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task
