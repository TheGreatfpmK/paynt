import importlib.util
import os
import sys

import pytest

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "finite-horizon-time-dependent.py")
LANDSCAPE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "finite-horizon-landscape.py")
DP_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "finite-horizon-dp.py")


@pytest.fixture(scope="module")
def fh():
    """The script, loaded as a module (its name is not a valid module name)."""
    spec = importlib.util.spec_from_file_location("finite_horizon_time_dependent", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look up the defining module in sys.modules
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture(scope="module")
def landscape(fh):
    """The script that draws the landscapes of the policies, as a module (it uses the solver of the fixture fh)."""
    spec = importlib.util.spec_from_file_location("finite_horizon_landscape", LANDSCAPE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture(scope="module")
def fhdp():
    """The exact search of finite-horizon-dp.py, as a module (it loads finite-horizon-time-dependent.py as its model layer)."""
    spec = importlib.util.spec_from_file_location("finite_horizon_dp", DP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]
