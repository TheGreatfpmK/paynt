#include "src/synthesis/synthesis.h"

#include "FiniteHorizon.h"

#include <pybind11/numpy.h>

#include <storm/exceptions/InvalidArgumentException.h>
#include <storm/storage/SparseMatrix.h>
#include <storm/utility/macros.h>

namespace synthesis {

/** Copy the vector to a numpy array. */
py::array_t<int64_t> toArray(std::vector<int64_t> const& vector) {
    return py::array_t<int64_t>(vector.size(), vector.data());
}

/**
 * Export the transition matrix in the CSR format, i.e. as (row pointers, column indices, values) numpy arrays. The row pointers of the matrix rows (not of
 * its row groups) are returned.
 */
std::tuple<py::array_t<int64_t>,py::array_t<int64_t>,py::array_t<double>> sparseMatrixToCsr(storm::storage::SparseMatrix<double> const& matrix) {
    uint64_t const num_rows = matrix.getRowCount();
    uint64_t const num_entries = matrix.getEntryCount();
    py::array_t<int64_t> row_pointers(num_rows+1);
    py::array_t<int64_t> columns(num_entries);
    py::array_t<double> values(num_entries);
    int64_t* row_pointers_data = row_pointers.mutable_data();
    int64_t* columns_data = columns.mutable_data();
    double* values_data = values.mutable_data();
    uint64_t position = 0;
    for(uint64_t row = 0; row < num_rows; ++row) {
        row_pointers_data[row] = position;
        for(auto const& entry: matrix.getRow(row)) {
            columns_data[position] = entry.getColumn();
            values_data[position] = entry.getValue();
            ++position;
        }
    }
    row_pointers_data[num_rows] = position;
    return std::make_tuple(row_pointers, columns, values);
}

/**
 * Export the coloring in the CSR format, i.e. as numpy arrays (choice pointers, holes, options): the color of the choice c is the list of pairs
 * (holes[i], options[i]) for i in [pointers[c],pointers[c+1]).
 */
std::tuple<py::array_t<int64_t>,py::array_t<int64_t>,py::array_t<int64_t>> coloringToCsr(Coloring const& coloring) {
    auto const& choice_to_assignment = coloring.getChoiceToAssignment();
    uint64_t const num_choices = choice_to_assignment.size();
    uint64_t num_pairs = 0;
    for(auto const& assignment: choice_to_assignment) {
        num_pairs += assignment.size();
    }
    py::array_t<int64_t> pointers(num_choices+1);
    py::array_t<int64_t> holes(num_pairs);
    py::array_t<int64_t> options(num_pairs);
    int64_t* pointers_data = pointers.mutable_data();
    int64_t* holes_data = holes.mutable_data();
    int64_t* options_data = options.mutable_data();
    uint64_t position = 0;
    for(uint64_t choice = 0; choice < num_choices; ++choice) {
        pointers_data[choice] = position;
        for(auto const& [hole,option]: choice_to_assignment[choice]) {
            holes_data[position] = hole;
            options_data[position] = option;
            ++position;
        }
    }
    pointers_data[num_choices] = position;
    return std::make_tuple(pointers, holes, options);
}

/** Construct a bit vector of the given size having exactly the given indices set. */
BitVector bitVectorFromIndices(uint64_t size, py::array_t<int64_t, py::array::c_style | py::array::forcecast> const& indices) {
    BitVector bit_vector(size, false);
    auto const* data = indices.data();
    for(py::ssize_t i = 0; i < indices.size(); ++i) {
        STORM_LOG_THROW(data[i] >= 0 and static_cast<uint64_t>(data[i]) < size, storm::exceptions::InvalidArgumentException, "index " << data[i] << " is out of range");
        bit_vector.set(data[i], true);
    }
    return bit_vector;
}

}

void bindings_finite_horizon(py::module& m) {

    py::classh<synthesis::FiniteHorizonUnfolder<double>>(m, "FiniteHorizonUnfolder", "Unfolds an MDP over a finite horizon, building only the reachable states.")
        .def(py::init<storm::models::sparse::Mdp<double> const&, uint64_t>(), py::keep_alive<1, 2>(), py::arg("mdp"), py::arg("horizon"))
        .def("construct_unfolded_model", &synthesis::FiniteHorizonUnfolder<double>::constructUnfoldedModel,
            "Construct the unfolded MDP; the states of the last layer are added to the given labels.", py::arg("final_layer_labels") = std::vector<std::string>()
        )
        .def_property_readonly("horizon", &synthesis::FiniteHorizonUnfolder<double>::getHorizon)
        .def_property_readonly("num_states", &synthesis::FiniteHorizonUnfolder<double>::numStates)
        .def_property_readonly("num_choices", &synthesis::FiniteHorizonUnfolder<double>::numChoices)
        .def_property_readonly("num_transitions", &synthesis::FiniteHorizonUnfolder<double>::numTransitions)
        .def_property_readonly("state_step", [](synthesis::FiniteHorizonUnfolder<double> const& u) { return synthesis::toArray(u.stateStep); })
        .def_property_readonly("state_base", [](synthesis::FiniteHorizonUnfolder<double> const& u) { return synthesis::toArray(u.stateBase); })
        .def_property_readonly("choice_step", [](synthesis::FiniteHorizonUnfolder<double> const& u) { return synthesis::toArray(u.choiceStep); })
        .def_property_readonly("choice_base", [](synthesis::FiniteHorizonUnfolder<double> const& u) { return synthesis::toArray(u.choiceBase); })
        ;
    m.def("unfoldFamilyFiniteHorizon", &synthesis::unfoldFamilyFiniteHorizon,
        "Unfold the family: a copy of each hole for every time step, the copy of hole h at step t has index t*K+h.",
        py::arg("family"), py::arg("horizon")
    );

    py::classh<synthesis::TimeUnfoldedColoring>(m, "TimeUnfoldedColoring", "Coloring extended with the time step, see FiniteHorizonUnfolder.")
        .def(py::init<synthesis::Coloring const&, synthesis::Family const&, synthesis::FiniteHorizonUnfolder<double> const&>(),
            py::arg("coloring"), py::arg("family"), py::arg("unfolder")
        )
        .def_property_readonly("horizon", &synthesis::TimeUnfoldedColoring::getHorizon)
        .def_property_readonly("num_holes", &synthesis::TimeUnfoldedColoring::numHoles)
        .def_property_readonly("num_choices", &synthesis::TimeUnfoldedColoring::numChoices)
        .def("getChoiceAssignment", &synthesis::TimeUnfoldedColoring::getChoiceAssignment)
        .def("selectCompatibleChoices", py::overload_cast<synthesis::Family const&>(&synthesis::TimeUnfoldedColoring::selectCompatibleChoices, py::const_))
        .def("selectCompatibleChoices",
            py::overload_cast<synthesis::Family const&, storm::storage::BitVector const&>(&synthesis::TimeUnfoldedColoring::selectCompatibleChoices, py::const_)
        )
        .def("collectHoleOptions", &synthesis::TimeUnfoldedColoring::collectHoleOptions)
        ;

    m.def("sparseMatrixToCsr", &synthesis::sparseMatrixToCsr,
        "Export the sparse matrix as numpy arrays (row pointers, column indices, values).", py::arg("matrix")
    );
    m.def("coloringToCsr", &synthesis::coloringToCsr,
        "Export the coloring as numpy arrays (choice pointers, holes, options).", py::arg("coloring")
    );
    m.def("bitVectorFromIndices", &synthesis::bitVectorFromIndices,
        "Construct a bit vector of the given size with the given indices set.", py::arg("size"), py::arg("indices")
    );
}
