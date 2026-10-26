#pragma once

#include "src/common.h"

#include <storm/adapters/RationalNumberAdapter.h>

#include <stdexcept>
#include <string>
#include <utility>

/**
 * Binding of class templates as generic families, the way stormpy binds its own (stormpy's src/template_binding.h is
 * not installed, so its registration format is replicated here). Each instantiation is bound under a native name such
 * as _PomdpManager_Double and recorded in the module's _template_instantiations table under its template arguments,
 * e.g. (float,); payntbind/synthesis/__init__.py exposes the recorded families via stormpy's _TemplateClass, e.g. as
 * PomdpManager[float], PomdpManager[stormpy.Rational], or PomdpManager(pomdp) with the value type taken from the model.
 *
 * Free function templates need none of this: their instantiations are bound as overloads of a single name and
 * pybind11 picks the one matching the arguments.
 */
namespace synthesis {

template <typename ValueType>
struct TemplateArgument;

template <>
struct TemplateArgument<double> {
    static py::object key() {
        return py::module_::import("builtins").attr("float");
    }
    static constexpr char const* label = "Double";
};

template <>
struct TemplateArgument<storm::RationalNumber> {
    static py::object key() {
        return py::module_::import("stormpy").attr("Rational");
    }
    static constexpr char const* label = "Rational";
};

/**
 * Bind the ValueType instantiation of the class template family as a class of module m.
 *
 * @tparam Options further py::classh options, e.g. a base class
 * @param extra further arguments of the py::classh constructor
 */
template <typename ValueType, typename Class, typename... Options, typename... Extra>
py::classh<Class, Options...> bindTemplateClass(py::module& m, std::string const& family, char const* doc,
                                                Extra&&... extra) {
    if (!py::hasattr(m, "_template_instantiations")) {
        m.attr("_template_instantiations") = py::dict();
    }
    py::dict families = m.attr("_template_instantiations");
    if (!families.contains(family)) {
        families[family.c_str()] = py::dict();
    }
    py::dict instantiations = families[family.c_str()];
    py::tuple key = py::make_tuple(TemplateArgument<ValueType>::key());
    if (instantiations.contains(key)) {
        throw std::runtime_error("duplicate instantiation of " + family + " for " + TemplateArgument<ValueType>::label);
    }
    std::string name = "_" + family + "_" + TemplateArgument<ValueType>::label;
    py::classh<Class, Options...> result(m, name.c_str(), doc, std::forward<Extra>(extra)...);
    instantiations[key] = result;
    return result;
}

}  // namespace synthesis
