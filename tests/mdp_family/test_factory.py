import paynt.colored_mdp
import paynt.mdp_family
import paynt.mdp_family._utils
import paynt.synthesizer.search_node


class TestFamilyColoredMdpFactory:

    def test_load_sketch_produces_a_family_colored_mdp(self, family_colored_mdp):
        assert type(family_colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert family_colored_mdp.feature_kind == "family"
        assert isinstance(family_colored_mdp.feature_info, paynt.mdp_family._utils.MdpFamilyInfo)

    def test_build_produces_an_mdp_not_necessarily_deterministic(self, family_colored_mdp):
        node = paynt.synthesizer.search_node.SearchNode(family_colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = family_colored_mdp.build(node.parameter_space)
        assert node.mdp.states > 0

    def test_build_assignment_does_not_force_a_dtmc(self, family_colored_mdp):
        """
        Regression test: ColoredMdp.build_assignment's "family"/"pomdp_family" branch specifically because
        fixing every parameter (picking one family member) does not fix the agent's policy -- the resulting
        model can still be nondeterministic, so it must not be converted to a DTMC the way it is for every
        other feature_kind.
        """
        assignment = family_colored_mdp.parameter_space.pick_any()
        mdp = family_colored_mdp.build_assignment(assignment)
        assert mdp.states > 0

    def test_empty_policy_has_one_entry_per_state(self, family_colored_mdp):
        policy = paynt.mdp_family._utils.empty_policy(family_colored_mdp)
        assert len(policy) == family_colored_mdp.underlying_mdp.nr_states
        assert all(action is None for action in policy)

    def test_action_structure_is_populated(self, family_colored_mdp):
        info = family_colored_mdp.feature_info
        assert info.num_actions > 0
        assert len(info.action_labels) == info.num_actions
        assert len(info.state_to_actions) == family_colored_mdp.underlying_mdp.nr_states
