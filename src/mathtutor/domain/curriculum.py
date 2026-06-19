from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from mathtutor.contracts import KnowledgeComponent


class CurriculumError(Exception):
    """Raised when curriculum invariants are violated."""


class Curriculum:
    """
    Directed acyclic prerequisite graph of knowledge components (KCs).

    Nodes are KCs keyed by `kc.id`.
    Edges point from prerequisite -> dependent.

    The graph is subject-agnostic: it stores only ids and prerequisite edges,
    so the same scheduling / mastery machinery teaches algebra, calculus, trig,
    probability, or anything else. Author a subject as data and load it with
    :meth:`from_json_file` / :meth:`from_json` / :meth:`from_dict`.
    """

    def __init__(
        self,
        kcs: List[KnowledgeComponent] | None = None,
        *,
        subject: str | None = None,
    ) -> None:
        self._kcs: Dict[str, KnowledgeComponent] = {}
        self.subject = subject

        if kcs:
            for kc in kcs:
                self.add(kc)

    @property
    def kcs(self) -> List[KnowledgeComponent]:
        """KCs in insertion order (prerequisite-shallow first).

        This is the public iterable that ``learner.scheduling.select_next`` and
        ``learner.bkt.propagate`` consume. They read only ``.id`` and
        ``.prerequisites`` from each element.
        """
        return list(self._kcs.values())

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

    # ------------------------------------------------------------------
    # Data-driven construction
    # ------------------------------------------------------------------
    #
    # Authoring format (JSON or an equivalent Python dict)
    # ----------------------------------------------------
    # Either a bare list of KC objects, or a wrapper with metadata:
    #
    #   {
    #     "subject": "calculus",
    #     "knowledge_components": [
    #       {
    #         "id": "limits_intro",               # REQUIRED, unique, non-empty
    #         "name": "Introduction to Limits",   # optional (defaults to id)
    #         "prerequisites": [],                # optional (defaults to [])
    #         "verifier_domain": "limits",        # optional (defaults to "")
    #         "difficulty_band": 1,               # optional int (defaults to 1)
    #         "generators": ["gen_limits_intro"]  # optional; string or list
    #       },
    #       ...
    #     ]
    #   }
    #
    # KCs may be listed in any order; the loader sorts them so each
    # prerequisite is inserted before its dependents, and raises
    # CurriculumError on a cycle, an unknown prerequisite, or a duplicate id.

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any] | List[Any],
        *,
        subject: str | None = None,
    ) -> "Curriculum":
        """Build a Curriculum from a parsed dict (or bare list of KC dicts)."""
        if isinstance(data, list):
            raw_kcs: Any = data
            meta_subject = subject
        elif isinstance(data, dict):
            raw_kcs = data.get("knowledge_components")
            if raw_kcs is None:
                raw_kcs = data.get("kcs")
            if raw_kcs is None:
                raise CurriculumError(
                    "Curriculum data must contain 'knowledge_components' (or 'kcs')."
                )
            meta_subject = subject or data.get("subject") or data.get("name")
        else:
            raise CurriculumError(
                f"Curriculum data must be a dict or list, got {type(data).__name__}."
            )

        if not isinstance(raw_kcs, list):
            raise CurriculumError("'knowledge_components' must be a list.")

        parsed = [cls._kc_from_dict(item, index=i) for i, item in enumerate(raw_kcs)]

        # Up-front validation gives clearer errors than add()'s incremental checks.
        ids = [kc.id for kc in parsed]
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        if dupes:
            raise CurriculumError(f"Duplicate KC id(s): {dupes}")

        id_set = set(ids)
        for kc in parsed:
            unknown = [p for p in kc.prerequisites if p not in id_set]
            if unknown:
                raise CurriculumError(
                    f"KC '{kc.id}' references unknown prerequisites: {unknown}"
                )

        # add() requires prerequisites to exist first, so order before inserting.
        ordered = cls._dependency_order(parsed)

        curriculum = cls(subject=meta_subject)
        for kc in ordered:
            curriculum.add(kc)
        return curriculum

    @classmethod
    def from_json(cls, text: str, *, subject: str | None = None) -> "Curriculum":
        """Build a Curriculum from a JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CurriculumError(f"Invalid JSON: {exc}") from exc
        return cls.from_dict(data, subject=subject)

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        subject: str | None = None,
    ) -> "Curriculum":
        """Build a Curriculum from a JSON file on disk.

        If no subject is given (and none is in the file), the file stem is used.
        """
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise CurriculumError(
                f"Could not read curriculum file {str(path)!r}: {exc}"
            ) from exc
        return cls.from_json(text, subject=subject or p.stem)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (round-trips through :meth:`from_dict`)."""
        return {
            "subject": self.subject,
            "knowledge_components": [
                {
                    "id": kc.id,
                    "name": kc.name,
                    "prerequisites": list(kc.prerequisites),
                    "verifier_domain": kc.verifier_domain,
                    "difficulty_band": kc.difficulty_band,
                    "generators": list(kc.generators),
                }
                for kc in self.kcs
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    # ------------------------------------------------------------------
    # Internal helpers for the loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _kc_from_dict(item: Any, *, index: int) -> KnowledgeComponent:
        """Validate one KC dict and build a contract-correct KnowledgeComponent."""
        if not isinstance(item, dict):
            raise CurriculumError(
                f"KC at position {index} must be an object, got {type(item).__name__}."
            )

        kc_id = item.get("id")
        if not isinstance(kc_id, str) or not kc_id.strip():
            raise CurriculumError(
                f"KC at position {index} is missing a non-empty string 'id'."
            )

        prereqs = item.get("prerequisites", [])
        if not isinstance(prereqs, list) or not all(isinstance(p, str) for p in prereqs):
            raise CurriculumError(
                f"KC '{kc_id}': 'prerequisites' must be a list of strings."
            )

        generators = item.get("generators", [])
        if isinstance(generators, str):
            generators = [generators]  # tolerate a single string for convenience
        if not isinstance(generators, list) or not all(
            isinstance(g, str) for g in generators
        ):
            raise CurriculumError(
                f"KC '{kc_id}': 'generators' must be a string or a list of strings."
            )

        band = item.get("difficulty_band", 1)
        # bool is a subclass of int — reject it explicitly.
        if not isinstance(band, int) or isinstance(band, bool):
            raise CurriculumError(
                f"KC '{kc_id}': 'difficulty_band' must be an integer."
            )

        name = item.get("name", kc_id)
        if not isinstance(name, str):
            raise CurriculumError(f"KC '{kc_id}': 'name' must be a string.")

        verifier_domain = item.get("verifier_domain", "")
        if not isinstance(verifier_domain, str):
            raise CurriculumError(
                f"KC '{kc_id}': 'verifier_domain' must be a string."
            )

        return KnowledgeComponent(
            id=kc_id,
            name=name,
            prerequisites=list(prereqs),
            verifier_domain=verifier_domain,
            difficulty_band=band,
            generators=list(generators),
        )

    @staticmethod
    def _dependency_order(
        kcs: List[KnowledgeComponent],
    ) -> List[KnowledgeComponent]:
        """Order KCs so every prerequisite precedes its dependents.

        Stable within each pass (preserves authoring order among independent
        KCs). Raises CurriculumError if no progress can be made, which means a
        cycle (prerequisites are guaranteed to exist by the caller).
        """
        inserted: Set[str] = set()
        ordered: List[KnowledgeComponent] = []
        remaining = list(kcs)

        while remaining:
            progressed = False
            next_remaining: List[KnowledgeComponent] = []
            for kc in remaining:
                if all(p in inserted for p in kc.prerequisites):
                    ordered.append(kc)
                    inserted.add(kc.id)
                    progressed = True
                else:
                    next_remaining.append(kc)
            remaining = next_remaining
            if not progressed:
                stuck = sorted(kc.id for kc in remaining)
                raise CurriculumError(
                    f"Cycle detected among knowledge components: {stuck}"
                )

        return ordered


def build_sample_curriculum() -> Curriculum:
    """
    Build a small sample curriculum:
    fractions -> equations -> quadratics

    Every KnowledgeComponent matches the contract in
    ``contracts.KnowledgeComponent``: ``generators`` is a list (plural) and
    ``difficulty_band`` is required.
    """
    curriculum = Curriculum(subject="intro_algebra")

    curriculum.add(
        KnowledgeComponent(
            id="fraction_basics",
            name="Fraction Basics",
            prerequisites=[],
            verifier_domain="fractions",
            difficulty_band=1,
            generators=["generate_fraction_basics"],
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="fraction_operations",
            name="Fraction Operations",
            prerequisites=["fraction_basics"],
            verifier_domain="fractions",
            difficulty_band=1,
            generators=["fraction_addition"],  # real registered generator
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="simplify_expressions",
            name="Simplify Expressions",
            prerequisites=["fraction_operations"],
            verifier_domain="algebraic_simplification",
            difficulty_band=2,
            generators=["generate_simplify_expressions"],
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="linear_one_step",
            name="One-Step Linear Equations",
            prerequisites=["simplify_expressions"],
            verifier_domain="linear_equations",
            difficulty_band=1,
            generators=["linear_equation"],  # real registered generator
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="linear_multi_step",
            name="Multi-Step Linear Equations",
            prerequisites=["linear_one_step"],
            verifier_domain="linear_equations",
            difficulty_band=2,
            generators=["generate_linear_multi_step"],
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="distributive_property",
            name="Distributive Property",
            prerequisites=["linear_one_step"],
            verifier_domain="expression_expansion",
            difficulty_band=2,
            generators=["generate_distributive_property"],
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="factoring_quadratics",
            name="Factoring Quadratics",
            prerequisites=["distributive_property", "linear_multi_step"],
            verifier_domain="quadratics",
            difficulty_band=3,
            generators=["generate_factoring_quadratics"],
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="solve_quadratics",
            name="Solve Quadratics",
            prerequisites=["factoring_quadratics"],
            verifier_domain="quadratics",
            difficulty_band=3,
            generators=["quadratic_equation"],  # real registered generator
        )
    )

    return curriculum


# ---------------------------------------------------------------------------
# Bundled curricula (works for editable installs AND installed wheels/zips)
# ---------------------------------------------------------------------------
#
# Curricula authored under src/mathtutor/data/curricula/*.json ship inside the
# package. These helpers read them through importlib.resources rather than a
# filesystem path, so loading works identically whether the package is run from
# a source checkout or installed as a wheel. For curricula stored OUTSIDE the
# package (e.g. a user-supplied file), use Curriculum.from_json_file(path).

_CURRICULA_ANCHOR = "mathtutor"             # the package that ships the data
_CURRICULA_SUBPATH = ("data", "curricula")  # ...data/curricula/<name>.json


def _curricula_dir():
    """Return a Traversable for the bundled curricula directory."""
    return importlib.resources.files(_CURRICULA_ANCHOR).joinpath(*_CURRICULA_SUBPATH)


def list_curricula() -> List[str]:
    """Return the names (without the .json suffix) of every bundled curriculum.

    Traverses package resources, so it works for source checkouts, installed
    wheels, and zipimport alike. Returns an empty list if the data directory is
    missing.
    """
    names: List[str] = []
    try:
        for entry in _curricula_dir().iterdir():
            if entry.name.endswith(".json"):
                names.append(entry.name[: -len(".json")])
    except (FileNotFoundError, NotADirectoryError, OSError):
        pass
    return sorted(names)


def load_curriculum(name: str) -> Curriculum:
    """Load a curriculum bundled under ``mathtutor/data/curricula`` by name.

    ``name`` may be given with or without the ``.json`` suffix::

        calc = load_curriculum("calculus_intro")

    The subject is set to the file stem (matching ``from_json_file``), so the
    curriculum reports ``subject == "calculus_intro"`` even if the file omits a
    ``subject`` field.

    Raises:
        CurriculumError: if no bundled curriculum has that name.
    """
    stem = name[: -len(".json")] if name.endswith(".json") else name
    resource = _curricula_dir().joinpath(f"{stem}.json")
    try:
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        available = list_curricula()
        hint = f" Available: {available}" if available else ""
        raise CurriculumError(f"No bundled curriculum named {stem!r}.{hint}") from exc
    return Curriculum.from_json(text, subject=stem)