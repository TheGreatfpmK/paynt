"""
Main implementation of Colored MDP (Definition 2 of arXiv:2511.08078): a colored MDP C = (M, V, kappa) is an MDP M, a
constrained parameter space V, and a coloring kappa: S x Act subseteq V such that for every theta in V
and state s in S it holds that there's exactly one action a such that theta in kappa(s, a).

Connection to old PAYNT implementation and explanation of new structure (TODO remove in the future):
This is the renamed, trimmed successor of paynt.quotient.quotient.Quotient. Every method here either
constructs an induced model via kappa (build/build_assignment, matching Definition 3's C[eta]/C[theta]) or
inverts kappa to interpret a scheduler (scheduler_selection/scheduler_is_consistent) -- i.e. everything
genuinely touches the coloring or the parameter space. Generic stormpy/index-space plumbing and numeric
model-checking-result facts that touch neither (restricting an MDP to a choice mask, converting between
scheduler/choice representations, choice values, expected visits) live in
paynt.model.model.ModelIndex; the one splitting heuristic that does need the coloring
(estimate_scheduler_difference) lives in paynt.utils.scoring instead, since it is search-algorithm support
shared by multiple synthesizer classes, not part of the representation.
"""

from __future__ import annotations

from typing import Any

import paynt.task
import paynt.parameter_space.parameter_space
import paynt.model.model
import paynt.synthesizer.search_node

import logging

logger = logging.getLogger(__name__)


class ColoredMdp:

    # discriminator used by dispatch code to pick the right feature package without an isinstance check;
    # concrete subclasses override this with their own feature name (e.g. "dt", "pomdp", "posmg", "family")
    feature_kind = "generic"

    # label associated with un-labelled choices, shared by every coloring-construction implementation
    EMPTY_LABEL = "__no_label__"

    def __init__(self, underlying_mdp: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, coloring: Any, use_exact: bool = False):
        # M: the underlying (uncolored) MDP, a stormpy sparse model
        self.underlying_mdp = underlying_mdp
        # V: the constrained parameter space
        self.parameter_space = parameter_space
        # kappa: raw payntbind coloring object (Coloring or ColoringSmt); no Python wrapper exists for this
        self.coloring = coloring
        self.use_exact = use_exact

        # internal plumbing needed by build()/scheduler_selection() below, not part of the public contract
        self.subsystem_builder_options = paynt.model.model.SubmodelBuilder.default_builder_options()
        self.choice_destinations = paynt.model.model.ModelIndex.compute_choice_destinations(underlying_mdp, use_exact)

    def export_result(self, dtmc: Any) -> None:
        """to be overridden"""

    def build(
        self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, parent_selected_choices: Any = None
    ) -> tuple[paynt.model.model.SubMdp, Any]:
        """
        Compute the induced sub-MDP C[eta] for the given parameter (sub)space.
        :param parent_selected_choices optional reuse hint: the parent search node's own selected_choices
            (its compatible-choices bitmask). Since parameter_space is always a narrowing of the parent's
            (a child never widens what its parent already assumed), any choice compatible with
            parameter_space must already have been compatible with the parent -- so restricting the search
            to parent_selected_choices is sound and cannot miss a choice, only skip ones already known
            incompatible with an ancestor. None for the search root, which has no parent to reuse.
        :returns (mdp, selected_choices)
        """
        if parent_selected_choices is None:
            choices = self.coloring.selectCompatibleChoices(parameter_space.native)
        else:
            choices = self.coloring.selectCompatibleChoices(parameter_space.native, parent_selected_choices)
        mdp = paynt.model.model.SubmodelBuilder.build_submdp(self.underlying_mdp, choices, self.subsystem_builder_options)
        mdp.parameter_space = parameter_space
        return mdp, choices

    def build_assignment(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> paynt.model.model.SubMdp:
        """Compute the induced DTMC C[theta] for a full parameter assignment."""
        assert parameter_space.size == 1, "expecting parameter space of size 1"
        choices = self.coloring.selectCompatibleChoices(parameter_space.native)
        assert choices.number_of_set_bits() > 0
        model, state_map, choice_map = paynt.model.model.SubmodelBuilder.restrict(self.underlying_mdp, choices, self.subsystem_builder_options)
        dtmc = paynt.model.model.SubmodelBuilder.mdp_to_dtmc(model)
        return paynt.model.model.SubMdp(dtmc, state_map, choice_map)

    def scheduler_selection(self, mdp: Any, scheduler: Any) -> list[list[int]]:
        """Get parameter options involved in the scheduler selection (the inverse of build(): choices -> V)."""
        assert scheduler.memoryless and scheduler.deterministic
        state_to_choice = paynt.model.model.ModelIndex.scheduler_to_state_to_choice(self.underlying_mdp, self.choice_destinations, mdp, scheduler)
        choices = paynt.model.model.ModelIndex.state_to_choice_to_choices(self.underlying_mdp, state_to_choice)
        return self.coloring.collectHoleOptions(choices)

    def scheduler_is_consistent(
        self, mdp: Any, node: paynt.synthesizer.search_node.SearchNode, result: Any, specification: Any
    ) -> tuple[list[list[int]], bool]:
        """
        Get the parameter assignment induced by this scheduler and fill undefined
        parameters by some option from the parameter space of this mdp.
        :param node the search node currently being verified -- unused by this base implementation, but part
            of the signature since DtColoredMdp's override needs it (to record scheduler_choices) and callers
            dispatch polymorphically without knowing which one they're calling
        :param specification the specification currently being solved for -- unused by this base
            implementation, but part of the signature since DtColoredMdp's override needs it and callers
            dispatch polymorphically without knowing which one they're calling
        :return parameter assignment
        :return whether the scheduler is consistent (i.e. corresponds to exactly one assignment)
        """
        if mdp.is_deterministic:
            selection = [[mdp.parameter_space.parameter_options(parameter)[0]] for parameter in range(mdp.parameter_space.num_parameters)]
            return selection, True

        # get qualitative scheduler selection, filter inconsistent assignments
        selection = self.scheduler_selection(mdp, result.scheduler)
        inconsistent_assignments = {parameter: options for parameter, options in enumerate(selection) if len(options) > 1}
        scheduler_is_consistent = len(inconsistent_assignments) == 0
        for parameter, options in enumerate(selection):
            if len(options) == 0:
                # if some parameter options are not involved in the selection, we can fix an arbitrary value
                selection[parameter] = [mdp.parameter_space.parameter_options(parameter)[0]]

        return selection, scheduler_is_consistent


class IdentityColoredMdpFactory:
    """
    Trivial "factory" for the plain generic ColoredMdp case (e.g. a DTMC with holes sketch): unlike every other
    feature, this ColoredMdp is built directly rather than through a real factory, and never needs
    re-unfolding at a different depth/memory size. Exists purely so Sketch.load_sketch/paynt.api.get_synthesizer
    can treat every feature uniformly as a (colored_mdp_factory, task) pair without special-casing this one.
    """

    def __init__(self, colored_mdp: ColoredMdp, task: paynt.task.SynthesisTask):
        self.colored_mdp = colored_mdp
        self.task = task
