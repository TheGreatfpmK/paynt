from __future__ import annotations

import dataclasses
from typing import Any

import stormpy
import payntbind

import paynt.model.model_builder
import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.mdp_family
import paynt.mdp_family.task
import paynt.posmg
import paynt.posmg.task
import paynt.pomdp
import paynt.pomdp.task
import paynt.specification.property
import paynt.task

from paynt.dt import DtColoredMdpFactory
import paynt.dt.dtnest.task

from paynt.parser.prism_parser import PrismParser
from paynt.parser.drn_parser import DrnParser
from paynt.parser.jani import JaniUnfolder

from ._utils import substitute_suffix, make_rewards_action_based

import os
import json

import logging

logger = logging.getLogger(__name__)


def _dataclass_task_kwargs(cls: type, task_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    load_sketch doesn't know which feature a sketch needs until it has parsed it, so task_kwargs carries
    every feature's CLI options at once; each feature task (DtTask, DtNestTask, PomdpTask, PosmgTask,
    FamilyTask) is a plain dataclass with no **kwargs catch-all of its own to swallow the rest (unlike
    SynthesisTask.from_specification, which still has one, since SynthesisTask isn't a dataclass) -- this
    keeps only the keys the target dataclass actually declares as real constructor (init=True) fields.
    """
    field_names = {f.name for f in dataclasses.fields(cls) if f.init}
    return {key: value for key, value in task_kwargs.items() if key in field_names}


class Sketch:

    @classmethod
    def load_sketch(
        cls,
        sketch_path: str,
        properties_path: str,
        export: str | None = None,
        relative_error: float = 0,
        precision: float = 1e-4,
        constraint_bound: Any = None,
        use_exact: bool = False,
        task_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, paynt.task.SynthesisTask]:

        # this function's real types are heavily branch-dependent (which of the PRISM/DRN/Cassandra parsers
        # ran, and, for PRISM, whether the sketch had parameters) -- kept as Any/Optional rather than forcing
        # every branch's read site to re-narrow a handful of mutually-exclusive control-flow paths
        prism: Any = None
        explicit_model: Any = None
        specification: paynt.specification.property.Specification | None = None
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace | None = None
        coloring: Any = None
        jani_unfolder: JaniUnfolder | None = None
        decpomdp_manager: Any = None
        obs_evaluator: Any = None

        paynt.specification.property.Property.model_checking_precision = precision

        # check path
        if not os.path.isfile(sketch_path):
            raise ValueError(f"the sketch file {sketch_path} does not exist")
        logger.info(f"loading sketch from {sketch_path} ...")

        filetype = None
        try:
            logger.info("assuming sketch in PRISM format...")
            prism, explicit_model, specification, parameter_space, coloring, jani_unfolder, obs_evaluator = PrismParser.read_prism(
                sketch_path, properties_path, relative_error, use_exact
            )
            filetype = "prism"
        except SyntaxError:
            pass
        if filetype is None:
            try:
                logger.info("assuming sketch in DRN format...")
                explicit_model = paynt.model.model_builder.ModelBuilder.from_drn(sketch_path, use_exact)
                specification = PrismParser.parse_specification(properties_path, relative_error, use_exact=use_exact)
                filetype = "drn"
                project_path = os.path.dirname(sketch_path)
                valuations_filename = "state-valuations.json"
                valuations_path = project_path + "/" + valuations_filename
                state_valuations = None
                if os.path.exists(valuations_path) and os.path.isfile(valuations_path):
                    with open(valuations_path) as file:
                        state_valuations = json.load(file)
                if state_valuations is not None:
                    if use_exact:
                        raise Exception("exact synthesis is not supported with state valuations")
                    logger.info(f"found state valuations in {valuations_path}, adding to the model...")
                    explicit_model = payntbind.synthesis.addStateValuations(explicit_model, state_valuations)
            except Exception as e:
                print(e)
        if filetype is None:
            try:
                logger.info("assuming sketch in Cassandra format...")
                decpomdp_manager = payntbind.synthesis.parse_decpomdp(sketch_path)
                if constraint_bound is not None:
                    decpomdp_manager.set_constraint(constraint_bound)
                if decpomdp_manager is None:
                    raise SyntaxError
                logger.info("applying discount factor transformation...")
                decpomdp_manager.apply_discount_factor_transformation()
                explicit_model = decpomdp_manager.construct_pomdp()
                if constraint_bound is not None:
                    specification = PrismParser.parse_specification(properties_path, relative_error)
                else:
                    optimality = paynt.specification.property.construct_reward_property(
                        decpomdp_manager.reward_model_name, decpomdp_manager.reward_minimizing, decpomdp_manager.discount_sink_label
                    )
                    specification = paynt.specification.property.Specification([optimality])
                filetype = "cassandra"
            except SyntaxError:
                pass

        assert filetype is not None, "unknown format of input file"
        assert specification is not None
        logger.info("sketch parsing OK")

        paynt.specification.property.Property.initialize(use_exact)
        if explicit_model.is_exact:
            updated = payntbind.synthesis.addMissingChoiceLabelsExact(explicit_model)
        else:
            updated = payntbind.synthesis.addMissingChoiceLabels(explicit_model)
        if updated is not None:
            explicit_model = updated
        if not payntbind.synthesis.assertChoiceLabelingIsCanonic(explicit_model.nondeterministic_choice_indices, explicit_model.choice_labeling, False):
            logger.warning("WARNING: choice labeling for the model is not canonic")

        make_rewards_action_based(explicit_model)
        logger.debug(f"constructed explicit model having {explicit_model.nr_states} states and {explicit_model.nr_choices} choices")

        if specification.contains_until_properties() and filetype != "prism":
            logger.info("WARNING: using until formulae with non-PRISM inputs might lead to unexpected behaviour")
        specification.transform_until_to_eventually()
        logger.info(f"found the following specification {specification}")

        if export is not None:
            Sketch.export(export, sketch_path, jani_unfolder, explicit_model)
            logger.info("export OK, aborting...")
            exit(0)

        task_kwargs = task_kwargs or {}
        task = paynt.task.SynthesisTask.from_specification(specification, use_exact=use_exact, **task_kwargs)

        colored_mdp_factory: Any
        build_task: Any
        if jani_unfolder is not None:
            assert parameter_space is not None
            if prism.model_type == stormpy.storage.PrismModelType.DTMC:
                colored_mdp = paynt.colored_mdp.ColoredMdp(explicit_model, parameter_space, coloring, use_exact=use_exact)
                colored_mdp_factory = paynt.colored_mdp.IdentityColoredMdpFactory(colored_mdp, task)
            elif prism.model_type == stormpy.storage.PrismModelType.MDP:
                build_task = paynt.mdp_family.task.FamilyTask(**_dataclass_task_kwargs(paynt.mdp_family.task.FamilyTask, task_kwargs))
                colored_mdp_factory = paynt.mdp_family.FamilyColoredMdpFactory(explicit_model, parameter_space, coloring, build_task, use_exact=use_exact)
            elif prism.model_type == stormpy.storage.PrismModelType.POMDP:
                build_task = paynt.mdp_family.task.FamilyTask(**_dataclass_task_kwargs(paynt.mdp_family.task.FamilyTask, task_kwargs))
                colored_mdp_factory = paynt.mdp_family.PomdpFamilyColoredMdpFactory(
                    explicit_model, parameter_space, coloring, build_task, obs_evaluator, use_exact=use_exact
                )
        else:
            # assert explicit_model.is_nondeterministic_model, "expected nondeterministic model"
            if decpomdp_manager is not None and decpomdp_manager.num_agents > 1:
                build_task = paynt.pomdp.task.PomdpTask(**_dataclass_task_kwargs(paynt.pomdp.task.PomdpTask, task_kwargs))
                colored_mdp_factory = paynt.pomdp.decpomdp.DecPomdpColoredMdpFactory(decpomdp_manager, build_task, use_exact=use_exact)
            elif isinstance(explicit_model, payntbind.synthesis.Posmg):
                build_task = paynt.posmg.task.PosmgTask(**_dataclass_task_kwargs(paynt.posmg.task.PosmgTask, task_kwargs))
                colored_mdp_factory = paynt.posmg.PosmgColoredMdpFactory(explicit_model, build_task, specification, use_exact=use_exact)
            elif not explicit_model.is_partially_observable:
                # always use the more capable DtNestTask (a strict superset of DtTask) since at
                # this point we don't yet know whether the caller intends to run dtnest or plain AR on this
                # sketch
                build_task = paynt.dt.dtnest.task.DtNestTask(**_dataclass_task_kwargs(paynt.dt.dtnest.task.DtNestTask, task_kwargs))
                colored_mdp_factory = DtColoredMdpFactory(explicit_model, build_task, use_exact=use_exact)
            else:
                build_task = paynt.pomdp.task.PomdpTask(**_dataclass_task_kwargs(paynt.pomdp.task.PomdpTask, task_kwargs))
                colored_mdp_factory = paynt.pomdp.PomdpColoredMdpFactory(explicit_model, build_task, decpomdp_manager, use_exact=use_exact)
        return colored_mdp_factory, task

    @classmethod
    def export(cls, export: str, sketch_path: str, jani_unfolder: JaniUnfolder | None, explicit_model: Any) -> None:
        if export == "jani":
            assert jani_unfolder is not None, "jani unfolder was not used"
            output_path = substitute_suffix(sketch_path, ".", "jani")
            jani_unfolder.write_jani(output_path)
        if export == "drn":
            output_path = substitute_suffix(sketch_path, ".", "drn")
            stormpy.export_to_drn(explicit_model, output_path)
        if export == "pomdp":
            assert explicit_model.is_nondeterministic_model and explicit_model.is_partially_observable, "cannot '--export pomdp' with non-POMDP sketches"
            output_path = substitute_suffix(sketch_path, ".", "pomdp")
            property_path = substitute_suffix(sketch_path, "/", "props.pomdp")
            DrnParser.write_model_in_pomdp_solve_format(explicit_model, output_path, property_path)
