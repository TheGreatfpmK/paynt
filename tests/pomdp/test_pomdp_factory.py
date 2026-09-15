import paynt.colored_mdp
import paynt.pomdp
import paynt.pomdp._utils
import paynt.synthesizer.search_node


class TestPomdpColoredMdpFactory:

    def test_factory_does_not_build_automatically(self, pomdp_colored_mdp_factory):
        """Guardrail for the "factories don't eagerly build" redesign: a freshly-constructed factory holds
        no colored_mdp/memory-size state of its own -- build()/set_imperfect_memory_size() must be called
        explicitly to get one."""
        assert not hasattr(pomdp_colored_mdp_factory, "colored_mdp")
        assert not hasattr(pomdp_colored_mdp_factory, "current_memory_size")

    def test_load_sketch_produces_a_pomdp_colored_mdp(self, pomdp_colored_mdp):
        assert type(pomdp_colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert pomdp_colored_mdp.feature_kind == "pomdp"
        assert isinstance(pomdp_colored_mdp.feature_info, paynt.pomdp._utils.PomdpInfo)

    def test_build_produces_an_mdp(self, pomdp_colored_mdp):
        node = paynt.synthesizer.search_node.SearchNode(pomdp_colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = pomdp_colored_mdp.build(node.parameter_space)
        assert node.mdp.states > 0

    def test_set_imperfect_memory_size_produces_a_fresh_colored_mdp(self, pomdp_colored_mdp_factory):
        reunfolded = pomdp_colored_mdp_factory.set_imperfect_memory_size(2)
        assert reunfolded.feature_kind == "pomdp"
