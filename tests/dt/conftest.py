import pytest

import paynt.parser.sketch

from helpers.helper import get_sketch_paths


@pytest.fixture
def dt_colored_mdp_factory():
    sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def dt_task():
    """The SynthesisTask sibling of dt_colored_mdp_factory -- Sketch.load_sketch returns (factory, task) as two separate objects since the
    SynthesisTask/feature-task split, so a fixture needing both loads the sketch itself rather than reaching into the factory (which now only carries the DT-
    specific DtTask)."""
    sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
    _, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def dt_colored_mdp(dt_colored_mdp_factory):
    return dt_colored_mdp_factory.build()
