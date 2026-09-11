import paynt.specification.property_result


class _FakeConstraintResult:
    def __init__(self, sat, primary_selection=None):
        self.sat = sat
        self.primary_selection = primary_selection


class _FakeOptimalityResult:
    def __init__(self, improves_optimum=None, value=None, can_improve=None, primary_selection=None):
        self.improves_optimum = improves_optimum
        self.value = value
        # can_improve/primary_selection are MdpOptimalityResult-shaped fields, only exercised by
        # TestUndecidedResults below -- improves_optimum/value are PropertyResult-shaped, only exercised by
        # TestAcceptingDtmc. Kept as one fake class (rather than two) since _spec_result below builds a real
        # MdpSpecificationResult either way and both call sites only ever set the fields they need.
        self.can_improve = can_improve
        self.primary_selection = primary_selection


def _spec_result(constraint_sats, optimality_result=None):
    """Builds an MdpSpecificationResult (a strict superset of SpecificationResult -- accepting_dtmc is
    inherited unchanged) so this one helper serves both TestAcceptingDtmc (base-class behavior) and
    TestUndecidedResults (MdpSpecificationResult-only undecided_results())."""
    spec_result = paynt.specification.property_result.MdpSpecificationResult()
    spec_result.constraints_result = paynt.specification.property_result.ConstraintsResult([_FakeConstraintResult(sat) for sat in constraint_sats])
    spec_result.optimality_result = optimality_result
    return spec_result


class TestAcceptingDtmc:
    """
    Regression coverage for SpecificationResult.accepting_dtmc's contract: it always returns a
    (accepting: bool, improving_value) tuple, never a bare bool. A caller that writes
    `if result.accepting_dtmc(spec):` instead of unpacking gets a tuple -- which is truthy in Python
    regardless of its contents, including (False, None) -- so the check always passes. This was a real bug
    in SynthesizerAR.check_specification (paynt/synthesizer/synthesizer_ar.py), silently marking a
    constraint's result "sat" and recording an "admissible_assignment" for any consistent scheduler
    regardless of whether the resulting DTMC actually satisfied the specification. It went undetected
    because it only executes when a specification has at least one explicit constraint (empty and skipped
    entirely for the optimality-only sketches this refactor's CLI regression battery has used), fixed by
    unpacking first: `accepting,_ = result.accepting_dtmc(spec); if accepting:`.
    """

    def test_unsat_constraints_return_false_and_must_not_be_used_as_a_bare_condition(self):
        result = _spec_result(constraint_sats=[False])
        assert result.accepting_dtmc(specification=None) == (False, None)
        # the actual regression: a non-empty tuple is always truthy, so `if result.accepting_dtmc(...):`
        # would incorrectly treat this unsatisfying result as accepting -- only the unpacked first
        # element may be used as the condition
        accepting, _ = result.accepting_dtmc(specification=None)
        assert accepting is False

    def test_sat_constraints_no_optimality_returns_true_and_none(self):
        result = _spec_result(constraint_sats=[True])
        assert result.accepting_dtmc(specification=None) == (True, None)

    def test_sat_constraints_optimality_not_improving_returns_false(self):
        """The scenario that actually triggers the bug in a real multi-constraint-with-optimality search:
        a different, already-found assignment locked in a better optimum, so a later, still-constraint-
        satisfying assignment no longer "improves" -- accepting_dtmc correctly reports this as not
        accepting, which the old unfixed caller would have silently overridden."""
        result = _spec_result(constraint_sats=[True], optimality_result=_FakeOptimalityResult(improves_optimum=False))
        accepting, improving_value = result.accepting_dtmc(specification=None)
        assert accepting is False
        assert improving_value is None

    def test_sat_constraints_optimality_improving_returns_true_and_value(self):
        result = _spec_result(constraint_sats=[True], optimality_result=_FakeOptimalityResult(improves_optimum=True, value=0.75))
        assert result.accepting_dtmc(specification=None) == (True, 0.75)


class TestUndecidedResults:
    """
    Regression coverage for MdpSpecificationResult.undecided_results() -- the multi-constraint counterpart to
    undecided_result() (paper Section 3.3, "AR for Feasibility Synthesis with Multiple Constraints"): returns
    every still-relevant property result (undecided constraints ascending, then optimality if it can still
    improve), not just one, since multi-constraint AR splitting (synthesizer_ar.split_parameter_space,
    dt.synthesizer_ar_dt.split_undecided_space) needs to compare every remaining property's own candidate
    against each other, not just act on a single arbitrarily-chosen one.
    """

    def test_single_undecided_constraint_no_optimality_matches_undecided_result(self):
        spec_result = _spec_result(constraint_sats=[True, None])
        undecided_constraint = spec_result.constraints_result.results[1]
        assert spec_result.undecided_results() == [undecided_constraint]
        # N=1 backward-compatibility guarantee: with exactly one relevant property, the plural method must
        # agree with the singular one already relied on elsewhere (e.g. DT's harmonize_inconsistent_scheduler
        # fallback, Hybrid's priority-node heuristic)
        assert spec_result.undecided_results() == [spec_result.undecided_result()]

    def test_multiple_undecided_constraints_returned_in_ascending_index_order(self):
        spec_result = _spec_result(constraint_sats=[None, True, None])
        c0 = spec_result.constraints_result.results[0]
        c2 = spec_result.constraints_result.results[2]
        assert spec_result.undecided_results() == [c0, c2]

    def test_optimality_appended_last_when_it_can_still_improve(self):
        opt = _FakeOptimalityResult(can_improve=True)
        spec_result = _spec_result(constraint_sats=[None], optimality_result=opt)
        c0 = spec_result.constraints_result.results[0]
        assert spec_result.undecided_results() == [c0, opt]

    def test_optimality_omitted_when_it_cannot_improve(self):
        opt = _FakeOptimalityResult(can_improve=False)
        spec_result = _spec_result(constraint_sats=[None], optimality_result=opt)
        c0 = spec_result.constraints_result.results[0]
        assert spec_result.undecided_results() == [c0]

    def test_no_undecided_constraints_optimality_only_matches_undecided_result(self):
        opt = _FakeOptimalityResult(can_improve=True)
        spec_result = _spec_result(constraint_sats=[True], optimality_result=opt)
        assert spec_result.undecided_results() == [opt]
        assert spec_result.undecided_results() == [spec_result.undecided_result()]
