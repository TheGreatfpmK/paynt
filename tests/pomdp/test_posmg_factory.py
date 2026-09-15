import paynt.colored_mdp
import paynt.pomdp.posmg
import paynt.pomdp.posmg._utils
import paynt.synthesizer.search_node


class TestPosmgColoredMdpFactory:

    def test_load_sketch_produces_a_posmg_colored_mdp(self, posmg_colored_mdp):
        assert type(posmg_colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert posmg_colored_mdp.feature_kind == "posmg"
        assert isinstance(posmg_colored_mdp.feature_info, paynt.pomdp.posmg._utils.PosmgInfo)

    def test_build_produces_an_mdp(self, posmg_colored_mdp):
        node = paynt.synthesizer.search_node.SearchNode(posmg_colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = posmg_colored_mdp.build(node.parameter_space)
        assert node.mdp.states > 0

    def test_create_smg_from_mdp_attaches_player_indications(self, posmg_colored_mdp):
        node = paynt.synthesizer.search_node.SearchNode(posmg_colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = posmg_colored_mdp.build(node.parameter_space)
        smg = paynt.pomdp.posmg._utils.create_smg_from_mdp(posmg_colored_mdp.feature_info, node.mdp)
        assert smg.states == node.mdp.states

    def test_set_imperfect_memory_size_produces_a_fresh_colored_mdp(self, posmg_colored_mdp_factory):
        reunfolded = posmg_colored_mdp_factory.set_imperfect_memory_size(2)
        assert reunfolded.feature_kind == "posmg"
        assert posmg_colored_mdp_factory.current_memory_size == 2
