import stormpy.developer.storage
from stormpy._template import _TemplateClass, _deduce_from_object

from . import _synthesis
from ._synthesis import *

# Class templates are bound once per value type (see src/template_binding.h) and exposed as generic families, like
# stormpy's own: the value type is given explicitly, e.g. PomdpManager[stormpy.Rational](pomdp), or taken from the model
# passed to the constructor, e.g. PomdpManager(pomdp). Function templates are plain overloads.
_deduce_from_model = _deduce_from_object(stormpy.developer.storage.parameters_of_model, keyword=("pomdp", "mdp", "posmg"))

PomdpManager = _TemplateClass("payntbind.synthesis.PomdpManager", _synthesis, parameters=("ValueType",), deduce=_deduce_from_model)
PomdpManagerAposteriori = _TemplateClass("payntbind.synthesis.PomdpManagerAposteriori", _synthesis, parameters=("ValueType",), deduce=_deduce_from_model)
SubPomdpBuilder = _TemplateClass("payntbind.synthesis.SubPomdpBuilder", _synthesis, parameters=("ValueType",), deduce=_deduce_from_model)
MemoryUnfolder = _TemplateClass("payntbind.synthesis.MemoryUnfolder", _synthesis, parameters=("ValueType",), deduce=_deduce_from_model)
Posmg = _TemplateClass("payntbind.synthesis.Posmg", _synthesis, parameters=("ValueType",))
PosmgManager = _TemplateClass("payntbind.synthesis.PosmgManager", _synthesis, parameters=("ValueType",), deduce=_deduce_from_model)
