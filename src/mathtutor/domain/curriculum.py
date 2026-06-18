from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from mathtutor.contracts import KnowledgeComponent


class CurriculumError(Exception):
    """Raised when curriculum invariants are violated."""


class Curriculum:
    """
    Directed acyclic prerequisite graph of knowledge components (KCs).

    Nodes are KCs keyed by `kc.id`.
    Edges point from prerequisite -> dependent.
    """

    def __init__(self, kcs: List[KnowledgeComponent] | None = None) -> None:
        self._kcs: Dict[str, KnowledgeComponent] = {}

        if kcs:
            for kc in kcs:
                self.add(kc)

    def add(self, kc: KnowledgeComponent) -> None:
        """
        Add a KC to the curriculum.

        Raises:
            CurriculumError: on duplicate ID, missing prerequisite, or cycle.
        """
        if kc.id in self._kcs:
            raise CurriculumError(f"Duplicate KC id: {kc.id}")

        missing = [p for p in kc.prerequisites if p not in self._kcs]
        if missing:
            raise CurriculumError(
                f"KC '{kc.id}' references missing prerequisites: {missing}"
            )

        self._kcs[kc.id] = kc

        if not self.is_dag():
            del self._kcs[kc.id]
            raise CurriculumError(f"Adding KC '{kc.id}' introduces a cycle")

    def get(self, kc_id: str) -> KnowledgeComponent:
        """Return a KC by ID."""
        return self._kcs[kc_id]

    def prerequisites(self, kc_id: str) -> List[str]:
        """Return prerequisite IDs for a KC."""
        return list(self.get(kc_id).prerequisites)

    def dependents(self, kc_id: str) -> List[str]:
        """Return IDs of KCs depending on the given KC."""
        return [
            other.id
            for other in self._kcs.values()
            if kc_id in other.prerequisites
        ]

    def is_dag(self) -> bool:
        """Return True if the graph is acyclic."""
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def dfs(node: str) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True

            visiting.add(node)
            for dep in self.prerequisites(node):
                if not dfs(dep):
                    return False
            visiting.remove(node)
            visited.add(node)
            return True

        return all(dfs(node) for node in self._kcs)

    def topological_order(self) -> List[str]:
        """
        Return a valid topological ordering.

        Raises:
            CurriculumError: if graph is cyclic.
        """
        indegree = {node: 0 for node in self._kcs}

        for kc in self._kcs.values():
            for prereq in kc.prerequisites:
                indegree[kc.id] += 1

        queue = [node for node, deg in indegree.items() if deg == 0]
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for dep in self.dependents(node):
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(self._kcs):
            raise CurriculumError("Graph contains cycle")

        return order

    def unmet_prerequisites(
        self,
        kc_id: str,
        mastered: Set[str],
    ) -> List[str]:
        """Return prerequisite IDs not yet mastered."""
        return [
            prereq
            for prereq in self.prerequisites(kc_id)
            if prereq not in mastered
        ]

    def ready_kcs(self, mastered: Set[str]) -> List[str]:
        """
        Return KCs whose prerequisites are fully mastered and which are not yet mastered.
        """
        ready = []
        for kc_id in self._kcs:
            if kc_id in mastered:
                continue
            if not self.unmet_prerequisites(kc_id, mastered):
                ready.append(kc_id)
        return ready


def build_sample_curriculum() -> Curriculum:
    """
    Build a small sample curriculum:
    fractions -> equations -> quadratics
    """
    curriculum = Curriculum()

    curriculum.add(
        KnowledgeComponent(
            id="fraction_basics",
            name="Fraction Basics",
            prerequisites=[],
            verifier_domain="fractions",
            generator="generate_fraction_basics",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="fraction_operations",
            name="Fraction Operations",
            prerequisites=["fraction_basics"],
            verifier_domain="fractions",
            generator="generate_fraction_operations",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="simplify_expressions",
            name="Simplify Expressions",
            prerequisites=["fraction_operations"],
            verifier_domain="algebraic_simplification",
            generator="generate_simplify_expressions",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="linear_one_step",
            name="One-Step Linear Equations",
            prerequisites=["simplify_expressions"],
            verifier_domain="linear_equations",
            generator="generate_linear_one_step",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="linear_multi_step",
            name="Multi-Step Linear Equations",
            prerequisites=["linear_one_step"],
            verifier_domain="linear_equations",
            generator="generate_linear_multi_step",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="distributive_property",
            name="Distributive Property",
            prerequisites=["linear_one_step"],
            verifier_domain="expression_expansion",
            generator="generate_distributive_property",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="factoring_quadratics",
            name="Factoring Quadratics",
            prerequisites=["distributive_property", "linear_multi_step"],
            verifier_domain="quadratics",
            generator="generate_factoring_quadratics",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="solve_quadratics",
            name="Solve Quadratics",
            prerequisites=["factoring_quadratics"],
            verifier_domain="quadratics",
            generator="generate_solve_quadratics",
        )
    )

    return curriculum
