import os
import subprocess
import sys
import textwrap

from helpers.helper import get_sketch_paths

_RUN_SCRIPT = textwrap.dedent("""
    import sys

    import paynt.parser.sketch
    import paynt.synthesizer.synthesizer

    sketch_path, props_path = sys.argv[1], sys.argv[2]
    colored_mdp_factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp_factory.build(), task, "smpmc")
    result = synthesizer.run()
    assert result.success, "expected a satisfying assignment to be found"
    """)


class TestSynthesizerSmpmcCli:
    """Subprocess regression test mirroring test_synthesizer_ar.py: like every non-AR engine, SynthesizerSMPMC.synthesize_one does not call
    resource_limit_reached() inside a single blocking solver.check(), so an external kill is the only reliable way to bound it."""

    def test_does_not_crash_on_threshold_property(self):
        sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny")
        result = subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, sketch_path, props_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_does_not_crash_on_optimality_property(self):
        sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny", props_name="optimality.props")
        result = subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, sketch_path, props_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_does_not_crash_on_family_model(self):
        sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
        result = subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, sketch_path, props_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr

    def test_robust_synthesis_exits_cleanly(self):
        """Regression: MBQI's fresh() propagators held terms through Z3 sub-contexts that Z3 had already freed, corrupting the heap at interpreter exit (see
        SmpmcPropagator.owner_ctx) -- only a separate process sees that, and not on every run, as it depends on the heap layout; test_robust.py checks the
        invariant itself."""
        project = os.path.dirname(get_sketch_paths("tests/mdp-family-rocks-4-2")[0])
        result = subprocess.run(
            [sys.executable, "-m", "paynt", project, "--method", "smpmc", "--constraint", "exists_forall", "--smpmc-verify-robust"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "robustness verified by AR: True" in result.stdout
