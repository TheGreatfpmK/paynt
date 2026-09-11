import paynt.synthesizer.synthesizer_ar
import paynt.synthesizer.search_node
import paynt.utils.scoring
import paynt.specification.property_result

import logging

logger = logging.getLogger(__name__)


class DtSearchNode(paynt.synthesizer.search_node.SearchNode):
    def __init__(self, parameter_space, parent_info=None):
        super().__init__(parameter_space, parent_info)
        # set only by DtColoredMdp.scheduler_is_consistent, and only for a single-property specification --
        # None otherwise, giving the "scheduler preserved across split" shortcut in verify_parameter_space a
        # real declared default rather than the getattr(..., None) it used to need
        self.scheduler_choices = None


class SynthesizerARDt(paynt.synthesizer.synthesizer_ar.SynthesizerAR):
    """
    AR specialized for decision-tree synthesis: splits by parameter kind (action/decision/variable) rather
    than by scored inconsistency variance, and adds harmonization (retrying an inconsistent scheduler
    selection against both directions of one parameter before giving up) plus a "scheduler preserved across
    split" shortcut specific to how DtColoredMdp.scheduler_is_consistent reports single-property results.
    This is the inner search engine; the outer DtSynthesizer (paynt.dt.synthesizer) constructs a fresh
    instance of this class for every tree depth it tries, mirroring the SynthesizerARStorm/SayntSynthesizer
    split.
    """

    search_node_type = DtSearchNode

    def __init__(self, colored_mdp, task):
        super().__init__(colored_mdp, task)
        self.counters_reset()

    @property
    def method_name(self):
        return "AR (decision tree)"

    def verify_parameter_selection(self, node, parameter_selection):
        spec = self.task.specification
        assignment = node.parameter_space.assume_options_copy(parameter_selection)
        dtmc = self.colored_mdp.build_assignment(assignment)
        res = dtmc.check_specification(spec)
        if not res.constraints_result.sat:
            return
        if not spec.has_optimality:
            node.analysis_result.improving_assignment = assignment
            node.analysis_result.can_improve = False
            return
        assignment_value = res.optimality_result.value
        if spec.optimality.improves_optimum(assignment_value):
            # logger.info(f"harmonization achieved value {res.optimality_result.value}")
            self.num_harmonization_succeeded += 1
            node.analysis_result.improving_assignment = assignment
            node.analysis_result.improving_value = assignment_value
            node.analysis_result.can_improve = True
            self.update_optimum(node)

    def harmonize_inconsistent_scheduler(self, node):
        """
        Try forcing the harmonizing parameter (the first one still locally inconsistent in some undecided
        property's own scheduler) to each of its two inconsistent options in turn, hoping one of the two
        concrete assignments happens to satisfy the whole spec outright. A pure-L(h) node -- every remaining
        property's own scheduler is already locally consistent, but they disagree with each other -- has
        nothing here to harmonize against, so this is a no-op in that case: harmonization is a speed-up
        attempt, not something split_undecided_space depends on succeeding.
        """
        self.num_harmonizations += 1
        result = next((r for r in node.analysis_result.undecided_results() if any(len(options) > 1 for options in r.primary_selection)), None)
        if result is None:
            return
        parameter_selection = result.primary_selection
        harmonizing_parameter = [parameter for parameter, options in enumerate(parameter_selection) if len(options) > 1][0]
        selection_1 = parameter_selection.copy()
        selection_1[harmonizing_parameter] = [selection_1[harmonizing_parameter][0]]
        selection_2 = parameter_selection.copy()
        selection_2[harmonizing_parameter] = [selection_2[harmonizing_parameter][1]]
        for selection in [selection_1, selection_2]:
            self.verify_parameter_selection(node, selection)

    def verify_parameter_space(self, node):
        self.num_parameter_spaces_considered += 1
        parent_selected_choices = node.parent_info.selected_choices if node.parent_info is not None else None
        node.mdp, node.selected_choices = self.colored_mdp.build(node.parameter_space, parent_selected_choices)

        self.stat.iteration(node.mdp)
        # scheduler_choices is only ever populated by DtColoredMdp.scheduler_is_consistent when the
        # specification is single-property (see split_undecided_space below) -- for a multi-property specification
        # it stays None on every node, so the "scheduler preserved" shortcut must be skipped rather than
        # assumed available, falling through to a real (slower, but correct) model-check instead.
        if node.parent_info is not None and node.parent_info.scheduler_choices is not None:
            for choice in node.parent_info.scheduler_choices:
                if not node.selected_choices[choice]:
                    break
            else:
                # scheduler preserved in the sub-parameter-space
                self.num_schedulers_preserved += 1
                node.analysis_result = node.parent_info.analysis_result
                node.scheduler_choices = node.parent_info.scheduler_choices
                consistent, parameter_selection = self.colored_mdp.are_choices_consistent(node.scheduler_choices, node.parameter_space)
                assert not consistent
                if node.analysis_result.optimality_result is None:
                    for constraint_res in node.analysis_result.constraints_result.results:
                        constraint_res.primary_selection = parameter_selection
                else:
                    node.analysis_result.optimality_result.primary_selection = parameter_selection
                return

        self.num_parameter_spaces_model_checked += 1
        self.check_specification(node)
        if not node.analysis_result.can_improve:
            return
        self.harmonize_inconsistent_scheduler(node)

    def build_unsat_result(self):
        spec_result = paynt.specification.property_result.MdpSpecificationResult()
        spec_result.constraints_result = paynt.specification.property_result.ConstraintsResult([])
        spec_result.optimality_result = paynt.specification.property_result.MdpOptimalityResult(None)
        spec_result.evaluate(None)
        spec_result.can_improve = False
        return spec_result

    def scheduler_scores(self, selection):
        """Decision-tree splitting heuristic: classify inconsistent parameters by kind (action/decision/
        variable) and pick one deterministically, rather than scoring by choice-value variance -- a
        genuinely different algorithm from the shared AR default, not a performance variant of it."""
        inconsistent_assignments = {parameter: options for parameter, options in enumerate(selection) if len(options) > 1}
        assert len(inconsistent_assignments) > 0, f"obtained selection with no inconsistencies: {selection}"
        inconsistent_action_parameters = [
            (parameter, options) for parameter, options in inconsistent_assignments.items() if self.colored_mdp.is_action_parameter[parameter]
        ]
        inconsistent_decision_parameters = [
            (parameter, options) for parameter, options in inconsistent_assignments.items() if self.colored_mdp.is_decision_parameter[parameter]
        ]
        inconsistent_variable_parameters = [
            (parameter, options) for parameter, options in inconsistent_assignments.items() if self.colored_mdp.is_variable_parameter[parameter]
        ]

        # choose one splitter
        splitter = None
        # try action or decision parameters first
        if len(inconsistent_action_parameters) > 0:
            splitter = inconsistent_action_parameters[0][0]
        elif len(inconsistent_decision_parameters) > 0:
            splitter = inconsistent_decision_parameters[0][0]
        else:
            splitter = inconsistent_variable_parameters[0][0]
        assert splitter is not None, "splitter not set"
        # force the score of the selected splitter
        return {splitter: 10}

    def split_undecided_space(self, node):
        mdp = node.mdp
        assert not mdp.is_deterministic

        cr = node.analysis_result.constraints_result
        assert cr is not None

        # (i) L(h): cross-constraint disagreement among undecided constraints whose OWN scheduler is already
        # fully consistent -- same priority as the generic AR path (paper https://www.jair.org/index.php/jair/article/view/16593 Section 3.3)
        candidates = [
            cr.results[i].primary_selection for i in cr.undecided_constraints if all(len(options) <= 1 for options in cr.results[i].primary_selection)
        ]
        disagreements = paynt.utils.scoring.compute_incompatibility_levels(candidates)

        if disagreements:
            splitter = paynt.utils.scoring.parameters_with_max_incompatibility(disagreements)[0]
            used_options = disagreements[splitter]
        else:
            # (ii) DT's own kind-based heuristic (scheduler_scores), applied to the first still-relevant
            # property that actually has a local inconsistency to split on
            result = next((r for r in node.analysis_result.undecided_results() if any(len(options) > 1 for options in r.primary_selection)), None)
            assert result is not None, "undecided DT node has neither cross-constraint disagreement nor any inconsistent property to split on"
            parameter_assignments = result.primary_selection
            scores = self.scheduler_scores(parameter_assignments)
            splitter = paynt.utils.scoring.parameters_with_max_score(scores)[0]
            used_options = parameter_assignments[splitter]

        if self.colored_mdp.is_action_parameter[splitter] or self.colored_mdp.is_decision_parameter[splitter]:
            assert len(used_options) > 1
            core_suboptions, other_suboptions = mdp.parameter_space.suboptions_enumerate(splitter, used_options)
        else:
            # Variable-kind parameters encode an ordered threshold comparison (e.g. "yellow <= k") -- unlike
            # action/decision parameters (arbitrary discrete choices), only a CONTIGUOUS sub-range of their
            # domain is a valid tree-branch predicate. suboptions_enumerate's singleton-per-value-plus-residual
            # shape produces a non-contiguous bucket (e.g. "value in {0,2,4}") the underlying coloring cannot
            # represent -- confirmed via a real crash (AssertionError: option not in the parameter space,
            # surfaced by dtnest on dt-orchard) when this branch was unified with the action/decision one
            # above. Generalizes the old exactly-2-options contiguous cut to N-ary L(h) disagreement instead:
            # sort the disagreeing values by their position in the parameter's full ordered domain and cut
            # just before each one except the first -- for exactly 2 values this reduces to precisely the old
            # cut (cutting once, at the second value's position), and the whole domain is always covered, with
            # no residual bucket.
            splitter_options = node.parameter_space.parameter_options(splitter)
            cut_positions = sorted(splitter_options.index(option) for option in used_options)[1:]
            boundaries = [0] + cut_positions + [len(splitter_options)]
            core_suboptions = [splitter_options[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
            for options in core_suboptions:
                assert len(options) > 0
            other_suboptions = []

        if len(other_suboptions) == 0:
            suboptions = core_suboptions
        else:
            suboptions = [other_suboptions] + core_suboptions  # DFS solves core first

        # construct corresponding child search nodes (SearchNode.split snapshots node's ParentInfo-relevant
        # state and hands it, shared, to every freshly-split child); layer the DT-specific parent_info
        # fields on top -- these are now always real declared fields (ParentInfo.analysis_result/
        # scheduler_choices), never a dynamic bolt-on that could silently be absent
        child_nodes = node.split(splitter, suboptions)
        assert node.parameter_space.size == sum([child.parameter_space.size for child in child_nodes])
        for child in child_nodes:
            child.parent_info.analysis_result = node.analysis_result
            # None (not just absent) for a multi-property specification -- see the guard in verify_parameter_space
            child.parent_info.scheduler_choices = node.scheduler_choices
        return child_nodes

    def counters_reset(self):
        self.num_parameter_spaces_considered = 0
        self.num_parameter_spaces_skipped = 0
        self.num_parameter_spaces_model_checked = 0
        self.num_schedulers_preserved = 0
        self.num_harmonizations = 0
        self.num_harmonization_succeeded = 0

    # TODO remove, debugging only
    def counters_print(self):
        logger.info(f"families considered: {self.num_parameter_spaces_considered}")
        logger.info(f"families skipped by construction: {self.num_parameter_spaces_skipped}")
        logger.info(f"families with schedulers preserved: {self.num_schedulers_preserved}")
        logger.info(f"families model checked: {self.num_parameter_spaces_model_checked}")
        logger.info(f"harmonizations attempted: {self.num_harmonizations}")
        logger.info(f"harmonizations succeeded: {self.num_harmonization_succeeded}")
