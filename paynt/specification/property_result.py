from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from paynt.specification.property import Property, OptimalityProperty, Specification


class PropertyResult:
    def __init__(self, prop: Property, result: Any, value: Any):
        self.result = result
        self.value = value
        self.sat = prop.satisfies_threshold(value)
        self.improves_optimum: bool | None = None if not isinstance(prop, OptimalityProperty) else prop.improves_optimum(value)

    def __str__(self) -> str:
        return str(self.value)


class ConstraintsResult:
    """
    A list of constraint results.
    Note: some results might be None (not evaluated).
    """

    # PropertyResult for a plain specification, MdpPropertyResult for the AR-family check_specification path
    # -- accessed generically here (only .sat), so kept as Any rather than a Union of the two unrelated classes
    def __init__(self, results: list[Any]):
        self.results = results
        self.undecided_constraints = [i for i, result in enumerate(results) if result is not None and result.sat is None]

        result_sats = [result.sat for result in results if result is not None]
        if False in result_sats:
            self.sat: bool | None = False
        elif None in result_sats:
            self.sat = None
        else:
            self.sat = True

    def __str__(self) -> str:
        return ",".join([str(result) for result in self.results])


@dataclass
class SpecificationResult:
    constraints_result: ConstraintsResult | None = None
    # PropertyResult for a plain specification, MdpOptimalityResult for an MdpSpecificationResult -- kept
    # as Any rather than a Union since the two concrete flows never mix on the same instance
    optimality_result: Any = None

    def __str__(self) -> str:
        return str(self.constraints_result) + " : " + str(self.optimality_result)

    def accepting_dtmc(self, specification: Specification) -> tuple[bool, Any | None]:
        """
        :return (1) whether specification is satisfied (dtmc is accepting)
        :return (2) new optimal value associated with the accepting dtmc
            (can be None if no optimality)
        """

        assert self.constraints_result is not None
        if not self.constraints_result.sat:
            # constraints not sat
            return False, None

        if self.optimality_result is None:
            # constraints sat and no optimality
            return True, None

        if not self.optimality_result.improves_optimum:
            # constraints sat and optimality not sat
            return False, None

        # constraints sat and optimality sat
        return True, self.optimality_result.value

    def undecided_result(self) -> Any:
        if self.optimality_result is not None and self.optimality_result.can_improve:
            return self.optimality_result
        assert self.constraints_result is not None
        return self.constraints_result.results[self.constraints_result.undecided_constraints[0]]


@dataclass
class MdpPropertyResult:
    prop: Property
    primary: Any = None
    secondary: Any = None
    sat: bool | None = None
    primary_selection: list[list[int]] | None = None

    @property
    def minimizing(self) -> bool:
        return self.prop.minimizing

    def __str__(self) -> str:
        prim = str(self.primary)
        seco = str(self.secondary)
        if self.minimizing:
            return f"{prim} - {seco}"
        return f"{seco} - {prim}"


@dataclass
class MdpOptimalityResult(MdpPropertyResult):
    improving_assignment: Any = None
    improving_value: Any = None
    can_improve: bool | None = None


@dataclass
class MdpSpecificationResult(SpecificationResult):
    improving_assignment: Any = None
    improving_value: Any = None
    can_improve: bool | None = None

    def undecided_results(self) -> list[Any]:
        """
        Every still-relevant property result: undecided constraints (ascending index), then optimality if
        it can still improve. Deliberately constraints-first/optimality-last -- the opposite priority from
        undecided_result()'s single optimality-first pick, which is a different use case (Hybrid's priority-
        node heuristic) and is kept as-is. Used by the multi-constraint AR splitting logic (paper
        https://www.jair.org/index.php/jair/article/view/16593 Section 3.3) to look at every property that
        could still inform which parameter to split on, not just one.
        """
        cr = self.constraints_result
        assert cr is not None
        results = [cr.results[i] for i in cr.undecided_constraints]
        if self.optimality_result is not None and self.optimality_result.can_improve:
            results.append(self.optimality_result)
        return results

    def evaluate(self, parameter_space: Any = None, admissible_assignment: Any = None) -> None:
        self.improving_assignment = None
        self.improving_value = None
        self.can_improve = None

        cr = self.constraints_result
        opt = self.optimality_result
        assert cr is not None

        if cr.sat is False:
            self.can_improve = False
            return

        if cr.sat is True:
            # all constraints were satisfied
            if opt is None:
                if admissible_assignment is not None:
                    self.improving_assignment = admissible_assignment
                else:
                    self.improving_assignment = parameter_space
                self.can_improve = False
            else:
                self.improving_assignment = opt.improving_assignment
                self.improving_value = opt.improving_value
                self.can_improve = opt.can_improve
            return

        # constraints undecided
        if opt is None:
            self.can_improve = True
        else:
            self.improving_assignment = opt.improving_assignment
            self.improving_value = opt.improving_value
            self.can_improve = opt.improving_value is None and opt.can_improve
