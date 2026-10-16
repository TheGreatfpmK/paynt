#include "FiniteHorizon.h"

#include <storm/exceptions/InvalidArgumentException.h>
#include <storm/exceptions/NotSupportedException.h>
#include <storm/models/sparse/ChoiceLabeling.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <storm/models/sparse/StateLabeling.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/storage/sparse/ModelComponents.h>
#include <storm/utility/constants.h>
#include <storm/utility/macros.h>

#include <algorithm>
#include <optional>

namespace synthesis {

template<typename ValueType>
FiniteHorizonUnfolder<ValueType>::FiniteHorizonUnfolder(storm::models::sparse::Mdp<ValueType> const& mdp, uint64_t horizon) : mdp(mdp), horizon(horizon) {
    STORM_LOG_THROW(horizon >= 1, storm::exceptions::InvalidArgumentException, "the horizon must be positive");

    auto const& matrix = mdp.getTransitionMatrix();
    auto const& row_groups = matrix.getRowGroupIndices();
    uint64_t const num_original_states = mdp.getNumberOfStates();

    // forward reachability, layer by layer
    layers.reserve(horizon + 1);
    layers.push_back(mdp.getInitialStates());
    for(uint64_t step = 0; step < horizon; ++step) {
        BitVector next(num_original_states, false);
        for(auto state: layers[step]) {
            for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
                num_choices++;
                for(auto const& entry: matrix.getRow(choice)) {
                    next.set(entry.getColumn(), true);
                    num_transitions++;
                }
            }
        }
        num_states += layers[step].getNumberOfSetBits();
        layers.push_back(std::move(next));
    }
    // the last layer: every state has a self-loop
    uint64_t const last_layer_size = layers[horizon].getNumberOfSetBits();
    num_states += last_layer_size;
    num_choices += last_layer_size;
    num_transitions += last_layer_size;
}

template<typename ValueType>
uint64_t FiniteHorizonUnfolder<ValueType>::getHorizon() const {
    return horizon;
}

template<typename ValueType>
uint64_t FiniteHorizonUnfolder<ValueType>::numStates() const {
    return num_states;
}

template<typename ValueType>
uint64_t FiniteHorizonUnfolder<ValueType>::numChoices() const {
    return num_choices;
}

template<typename ValueType>
uint64_t FiniteHorizonUnfolder<ValueType>::numTransitions() const {
    return num_transitions;
}

template<typename ValueType>
std::shared_ptr<storm::models::sparse::Mdp<ValueType>> FiniteHorizonUnfolder<ValueType>::constructUnfoldedModel(
    std::vector<std::string> const& final_layer_labels
) {
    auto const& matrix = mdp.getTransitionMatrix();
    auto const& row_groups = matrix.getRowGroupIndices();
    uint64_t const num_original_states = mdp.getNumberOfStates();

    storm::storage::sparse::ModelComponents<ValueType> components;

    // the states and their numbering: layer by layer
    std::vector<uint64_t> layer_offset(horizon + 2, 0);
    for(uint64_t step = 0; step <= horizon; ++step) {
        layer_offset[step+1] = layer_offset[step] + layers[step].getNumberOfSetBits();
    }
    stateStep.clear();
    stateBase.clear();
    stateStep.reserve(num_states);
    stateBase.reserve(num_states);
    for(uint64_t step = 0; step <= horizon; ++step) {
        for(auto state: layers[step]) {
            stateStep.push_back(step);
            stateBase.push_back(state);
        }
    }

    // transition matrix: the rows come out sorted, since the states of each layer are numbered in the order of the original states
    choiceStep.clear();
    choiceBase.clear();
    choiceStep.reserve(num_choices);
    choiceBase.reserve(num_choices);
    storm::storage::SparseMatrixBuilder<ValueType> builder(num_choices, num_states, num_transitions, true, true, num_states);
    std::vector<uint64_t> next_layer_index(num_original_states);
    uint64_t row = 0;
    for(uint64_t step = 0; step < horizon; ++step) {
        uint64_t index = layer_offset[step+1];
        for(auto state: layers[step+1]) {
            next_layer_index[state] = index++;
        }
        for(auto state: layers[step]) {
            builder.newRowGroup(row);
            for(uint64_t choice = row_groups[state]; choice < row_groups[state+1]; ++choice) {
                for(auto const& entry: matrix.getRow(choice)) {
                    builder.addNextValue(row, next_layer_index[entry.getColumn()], entry.getValue());
                }
                choiceStep.push_back(step);
                choiceBase.push_back(choice);
                ++row;
            }
        }
    }
    uint64_t index = layer_offset[horizon];
    for([[maybe_unused]] auto state: layers[horizon]) {
        builder.newRowGroup(row);
        builder.addNextValue(row, index, storm::utility::one<ValueType>());
        choiceStep.push_back(horizon);
        choiceBase.push_back(-1);
        ++row;
        ++index;
    }
    components.transitionMatrix = builder.build();

    // state labeling
    storm::models::sparse::StateLabeling labeling(num_states);
    for(auto const& label: mdp.getStateLabeling().getLabels()) {
        BitVector label_states(num_states, false);
        if(label == "init") {
            // the first layer consists of the initial states
            for(uint64_t unfolded_state = 0; unfolded_state < layer_offset[1]; ++unfolded_state) {
                label_states.set(unfolded_state, true);
            }
        } else {
            auto const& original_label_states = mdp.getStateLabeling().getStates(label);
            for(uint64_t unfolded_state = 0; unfolded_state < num_states; ++unfolded_state) {
                if(original_label_states[stateBase[unfolded_state]]) {
                    label_states.set(unfolded_state, true);
                }
            }
        }
        labeling.addLabel(label, std::move(label_states));
    }
    for(auto const& label: final_layer_labels) {
        if(not labeling.containsLabel(label)) {
            labeling.addLabel(label);
        }
        for(uint64_t unfolded_state = layer_offset[horizon]; unfolded_state < num_states; ++unfolded_state) {
            labeling.addLabelToState(label, unfolded_state);
        }
    }
    components.stateLabeling = std::move(labeling);

    // choice labeling
    if(mdp.hasChoiceLabeling()) {
        storm::models::sparse::ChoiceLabeling choice_labeling(num_choices);
        for(auto const& label: mdp.getChoiceLabeling().getLabels()) {
            auto const& original_label_choices = mdp.getChoiceLabeling().getChoices(label);
            BitVector label_choices(num_choices, false);
            for(uint64_t unfolded_choice = 0; unfolded_choice < num_choices; ++unfolded_choice) {
                if(choiceBase[unfolded_choice] >= 0 and original_label_choices[choiceBase[unfolded_choice]]) {
                    label_choices.set(unfolded_choice, true);
                }
            }
            choice_labeling.addLabel(label, std::move(label_choices));
        }
        components.choiceLabeling = std::move(choice_labeling);
    }

    // reward models
    for(auto const& [name,reward_model]: mdp.getRewardModels()) {
        STORM_LOG_THROW(
            not reward_model.hasStateRewards() and not reward_model.hasTransitionRewards(), storm::exceptions::NotSupportedException,
            "only state-action rewards are supported, reward model '" << name << "' has other types of rewards."
        );
        std::optional<std::vector<ValueType>> unfolded_rewards;
        if(reward_model.hasStateActionRewards()) {
            auto const& rewards = reward_model.getStateActionRewardVector();
            std::vector<ValueType> rewards_unfolded(num_choices, storm::utility::zero<ValueType>());
            for(uint64_t unfolded_choice = 0; unfolded_choice < num_choices; ++unfolded_choice) {
                if(choiceBase[unfolded_choice] >= 0) {
                    rewards_unfolded[unfolded_choice] = rewards[choiceBase[unfolded_choice]];
                }
            }
            unfolded_rewards = std::move(rewards_unfolded);
        }
        components.rewardModels.emplace(name, storm::models::sparse::StandardRewardModel<ValueType>(std::optional<std::vector<ValueType>>(), std::move(unfolded_rewards)));
    }

    return std::make_shared<storm::models::sparse::Mdp<ValueType>>(std::move(components));
}

Family unfoldFamilyFiniteHorizon(Family const& family, uint64_t horizon) {
    Family unfolded_family;
    for(uint64_t step = 0; step < horizon; ++step) {
        for(uint64_t hole = 0; hole < family.numHoles(); ++hole) {
            unfolded_family.addHole(family.holeNumOptionsTotal(hole));
        }
    }
    return unfolded_family;
}


TimeUnfoldedColoring::TimeUnfoldedColoring(Coloring const& base_coloring, Family const& base_family, FiniteHorizonUnfolder<double> const& unfolder)
    : base_choice_to_assignment(base_coloring.getChoiceToAssignment()), choice_step(unfolder.choiceStep), choice_base(unfolder.choiceBase),
      horizon(unfolder.getHorizon()) {
    STORM_LOG_THROW(choice_step.size() == unfolder.numChoices(), storm::exceptions::InvalidArgumentException, "the unfolded model has not been constructed yet");
    base_hole_num_options.reserve(base_family.numHoles());
    for(uint64_t hole = 0; hole < base_family.numHoles(); ++hole) {
        base_hole_num_options.push_back(base_family.holeNumOptionsTotal(hole));
    }
}

uint64_t TimeUnfoldedColoring::getHorizon() const {
    return horizon;
}

uint64_t TimeUnfoldedColoring::numBaseHoles() const {
    return base_hole_num_options.size();
}

uint64_t TimeUnfoldedColoring::numHoles() const {
    return horizon*numBaseHoles();
}

uint64_t TimeUnfoldedColoring::numChoices() const {
    return choice_step.size();
}

std::vector<std::pair<uint64_t,uint64_t>> TimeUnfoldedColoring::getChoiceAssignment(uint64_t choice) const {
    STORM_LOG_THROW(choice < numChoices(), storm::exceptions::InvalidArgumentException, "choice " << choice << " does not exist");
    if(choice_base[choice] < 0) {
        return {};
    }
    uint64_t const hole_offset = choice_step[choice]*numBaseHoles();
    auto assignment = base_choice_to_assignment[choice_base[choice]];
    for(auto& hole_option: assignment) {
        hole_option.first += hole_offset;
    }
    return assignment;
}

BitVector TimeUnfoldedColoring::selectCompatibleChoices(Family const& subfamily) const {
    return selectCompatibleChoices(subfamily, BitVector(numChoices(), true));
}

BitVector TimeUnfoldedColoring::selectCompatibleChoices(Family const& subfamily, BitVector const& base_choices) const {
    uint64_t const num_base_holes = numBaseHoles();
    BitVector selection(numChoices(), false);
    for(uint64_t choice = 0; choice < numChoices(); ++choice) {
        // uncolored choices carry nothing for any subfamily to exclude them on
        if(choice_base[choice] < 0 or base_choice_to_assignment[choice_base[choice]].empty()) {
            selection.set(choice, true);
            continue;
        }
        if(not base_choices[choice]) {
            continue;
        }
        uint64_t const hole_offset = choice_step[choice]*num_base_holes;
        bool compatible = true;
        for(auto const& [hole,option]: base_choice_to_assignment[choice_base[choice]]) {
            if(not subfamily.holeContains(hole_offset + hole, option)) {
                compatible = false;
                break;
            }
        }
        if(compatible) {
            selection.set(choice, true);
        }
    }
    return selection;
}

std::vector<std::vector<uint64_t>> TimeUnfoldedColoring::collectHoleOptions(BitVector const& choices) const {
    uint64_t const num_base_holes = numBaseHoles();
    std::vector<BitVector> hole_options_mask(numHoles());
    for(uint64_t step = 0; step < horizon; ++step) {
        for(uint64_t hole = 0; hole < num_base_holes; ++hole) {
            hole_options_mask[step*num_base_holes + hole] = BitVector(base_hole_num_options[hole], false);
        }
    }
    for(auto choice: choices) {
        if(choice_base[choice] < 0) {
            continue;
        }
        uint64_t const hole_offset = choice_step[choice]*num_base_holes;
        for(auto const& [hole,option]: base_choice_to_assignment[choice_base[choice]]) {
            hole_options_mask[hole_offset + hole].set(option, true);
        }
    }
    std::vector<std::vector<uint64_t>> hole_options(numHoles());
    for(uint64_t hole = 0; hole < numHoles(); ++hole) {
        for(auto option: hole_options_mask[hole]) {
            hole_options[hole].push_back(option);
        }
    }
    return hole_options;
}


// only the floating-point variant is instantiated (and bound): exact arithmetic is not used by any caller so far
template class FiniteHorizonUnfolder<double>;

}
