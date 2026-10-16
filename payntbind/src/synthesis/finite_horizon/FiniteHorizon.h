#pragma once

#include "src/synthesis/quotient/Coloring.h"
#include "src/synthesis/quotient/Family.h"

#include <storm/adapters/RationalNumberAdapter.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/storage/BitVector.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace synthesis {

using BitVector = storm::storage::BitVector;

/**
 * Unfold an MDP over a finite horizon, building only the states that are reachable.
 *
 * The unfolded MDP consists of the states (s,t), where t in [0,horizon] and s is a state of the original MDP that can be reached from an initial state in
 * exactly t steps. The states are numbered layer by layer (by t) and within a layer by the index of s. For t < horizon, the choices of (s,t) are the choices of s
 * (in the same order), leading to the states (s',t+1) with the original probabilities. Every state (s,horizon) of the last layer has a single choice, a
 * zero-reward self-loop. Labels, choice labels and state-action rewards are copied from the original MDP, with the exception of label "init", which only the
 * states of the first layer (i.e. the initial states) have. States of the last layer are additionally added to each of the final layer labels (a missing label is
 * created), which is how a caller makes running out of the horizon count as reaching a target.
 *
 * Which unfolded state or choice copies which state or choice of the original MDP is given by the maps below.
 * @note state rewards, transition rewards and state valuations are not supported (state valuations are dropped)
 */
template<typename ValueType>
class FiniteHorizonUnfolder {
public:

    /** Compute the reachable states, which fixes the size of the unfolded MDP. */
    FiniteHorizonUnfolder(storm::models::sparse::Mdp<ValueType> const& mdp, uint64_t horizon);

    uint64_t getHorizon() const;
    /** Number of states, choices and transitions of the unfolded MDP. */
    uint64_t numStates() const;
    uint64_t numChoices() const;
    uint64_t numTransitions() const;

    std::shared_ptr<storm::models::sparse::Mdp<ValueType>> constructUnfoldedModel(std::vector<std::string> const& final_layer_labels);

    /** For each state of the unfolded MDP, its step t. */
    std::vector<int64_t> stateStep;
    /** For each state of the unfolded MDP, the state s of the original MDP it copies. */
    std::vector<int64_t> stateBase;
    /** For each choice of the unfolded MDP, its step t (equal to the horizon for the self-loops of the last layer). */
    std::vector<int64_t> choiceStep;
    /** For each choice of the unfolded MDP, the choice of the original MDP it copies (-1 for the self-loops of the last layer). */
    std::vector<int64_t> choiceBase;

private:

    storm::models::sparse::Mdp<ValueType> const& mdp;
    uint64_t horizon;
    /** For each step, the states of the original MDP that are reachable in that many steps. */
    std::vector<BitVector> layers;
    uint64_t num_states = 0;
    uint64_t num_choices = 0;
    uint64_t num_transitions = 0;
};

/**
 * Unfold a family over a finite horizon: the unfolded family contains a copy of each of the original holes for every time step t in [0,horizon-1]. The copy
 * of the hole h at the step t has index t*K+h, where K is the number of holes of the original family.
 */
Family unfoldFamilyFiniteHorizon(Family const& family, uint64_t horizon);

/**
 * Coloring of the MDP unfolded by FiniteHorizonUnfolder wrt. the unfolded family of unfoldFamilyFiniteHorizon. The color (h,o) of the choice (c,t) becomes the
 * color (t*K+h,o), i.e. the original parameters are extended with the time step, which is why colors of choices in different time steps never conflict and an
 * inconsistency can occur only between choices of the same step. Choices of the last layer are uncolored.
 *
 * Unlike Coloring, whose memory requirement is quadratic in the horizon (it stores a mask over all holes for every choice and every state), this class
 * keeps just the coloring of the original MDP and the steps of the choices and so scales linearly. It implements the part of the Coloring interface that is
 * needed to select choices and to collect options of a parameter assignment, but not the rest of the interface (e.g. getStateToHoles()).
 */
class TimeUnfoldedColoring {
public:
    /** The unfolder must have constructed the unfolded model, since the choices of the coloring are the ones of that model. */
    TimeUnfoldedColoring(Coloring const& base_coloring, Family const& base_family, FiniteHorizonUnfolder<double> const& unfolder);

    uint64_t getHorizon() const;
    uint64_t numHoles() const;
    uint64_t numChoices() const;

    /** Get the color of an unfolded choice, as a list of (unfolded hole, option) pairs. */
    std::vector<std::pair<uint64_t,uint64_t>> getChoiceAssignment(uint64_t choice) const;

    /** Get a mask of choices compatible with the family. Uncolored choices (incl. the whole last layer) are always included. */
    BitVector selectCompatibleChoices(Family const& subfamily) const;
    /** Same as above, but a colored choice is selected only if it is also contained in \p base_choices. */
    BitVector selectCompatibleChoices(Family const& subfamily, BitVector const& base_choices) const;
    /** For each (unfolded) hole, collect options (colors) involved in any of the given choices. */
    std::vector<std::vector<uint64_t>> collectHoleOptions(BitVector const& choices) const;

protected:

    /** For each choice of the original MDP, a list of hole-option pairs (colors). */
    std::vector<std::vector<std::pair<uint64_t,uint64_t>>> base_choice_to_assignment;
    /** For each hole of the original family, the total number of its options. */
    std::vector<uint64_t> base_hole_num_options;
    /** For each choice of the unfolded MDP, its step and the choice of the original MDP (-1 if there is none). */
    std::vector<int64_t> choice_step;
    std::vector<int64_t> choice_base;
    uint64_t horizon;

    uint64_t numBaseHoles() const;
};

}
