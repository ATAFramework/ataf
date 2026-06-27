"""Tool builder — validates proposed code and turns it into a deployed tool.

This module is the **interesting bit** of the ATAF server. It takes the
raw Python source an LLM proposed inside ``<ATAF param="newtool">`` tags
and walks it through the deploy pipeline (see DESIGN.md §7):

    1. Parse + validate     — exactly one top-level function, fully
                              type-annotated, with a docstring, importing
                              only the Python stdlib (v0.1 restriction).
    2. Generate schema      — derive a JSON input schema from the function
                              signature and Google-style docstring.
    3. Persist              — allocate a ``tool_id`` and write code + row
                              through the registry.
    4. Return a BuildResult — the caller (``main.py``) uses this to build
                              the wire response and hot-register the route.

What this module deliberately does **not** do:

  * It does not acquire the global build lock. The caller is responsible
    for holding ``BuildLock`` across the whole build so that ``tool_id``
    allocation and insertion are serialized server-wide. This keeps the
    builder a pure, synchronously-testable unit.
  * It does not register the live FastAPI route. That needs the running
    ``app`` object and belongs in ``main.py``; mixing it in here would
    make the builder impossible to unit-test without a server.
  * It does not execute the proposed code. Validation is purely static
    (AST-based) so we never ``exec`` untrusted source at build time. The
    executor runs the code later, in a subprocess, only after a human has
    approved the tool.

Schema generation is hand-rolled from ``ast`` + a small Google-style
docstring parser rather than pulling in ``griffe`` or ``docstring-parser``
(plan.txt open question #2, resolved: hand-roll for v0.1, fewer deps).
"""

from __future__ import annotations

import ast
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from .eventlog import DeploymentEventLog
from .registry import Registry


# ----------------------------------------------------------------------
# Validation error — surfaced as 400 Bad Request at the API boundary.
# ----------------------------------------------------------------------
class CodeValidationError(Exception):
    """Raised when proposed tool code fails any validation rule.

    Carries a machine-readable ``code`` so ``main.py`` can put a stable
    error identifier in the wire response, plus a human-readable message
    that is safe to surface back to the LLM as a tool-result.

    Attributes:
        code: Stable error code, e.g. ``"MISSING_DOCSTRING"``,
            ``"NON_STDLIB_IMPORT"``, ``"MISSING_TYPE_ANNOTATION"``.
        message: Human-readable explanation of what was wrong.
    """

    def __init__(self, code: str, message: str) -> None:
        """Build the error.

        Args:
            code: Stable machine-readable error code.
            message: Human-readable explanation.
        """

        # Store both halves and hand the message up to Exception so that
        # str(err) is still useful in logs and tracebacks.
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# ----------------------------------------------------------------------
# Deployment error — surfaced as 500 Internal Server Error.
# ----------------------------------------------------------------------
class DeploymentError(Exception):
    """Raised when valid code passed validation but could not be deployed.

    This is deliberately distinct from ``CodeValidationError``. The
    submitted code was well-formed and met every rule — the failure was
    on the *server* side: a filesystem write, a SQLite insert, or
    ``tool_id`` allocation went wrong. Because it is the server's fault
    and not the agent's, the same code may well deploy successfully on a
    later retry once the underlying condition (disk full, locked DB, …)
    clears.

    The caller maps this to a ``500`` response carrying the stable code
    ``"TOOL_NOT_DEPLOYED"``.

    Attributes:
        code: Always ``"TOOL_NOT_DEPLOYED"``. Fixed (class-level) so the
            wire contract is stable regardless of the underlying cause.
        message: Human-readable explanation of what went wrong, derived
            from the underlying exception.
    """

    code = "TOOL_NOT_DEPLOYED"

    def __init__(self, message: str) -> None:
        """Build the error.

        Args:
            message: Human-readable explanation, usually ``str(cause)``.
        """

        # Keep the message addressable, and pass a prefixed form up to
        # Exception so tracebacks and logs are self-describing.
        self.message = message
        super().__init__(f"{self.code}: {message}")


# ----------------------------------------------------------------------
# Parsed-tool intermediate — the validated, structured view of the code.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _ParameterSpec:
    """One function parameter, extracted from the AST.

    Attributes:
        name: The parameter's identifier.
        annotation: The raw AST annotation node (e.g. an ``ast.Name`` for
            ``float``). Schema generation maps this to a JSON type.
        required: True if the parameter has no default value. Parameters
            with defaults become optional in the JSON schema.
    """

    name: str
    annotation: ast.expr
    required: bool


@dataclass(frozen=True)
class _ParsedTool:
    """The validated, structured form of a proposed tool's source.

    Produced by ``Builder._validate()`` and consumed by the schema
    generator. Not part of the public API — callers see ``BuildResult``.

    Attributes:
        name: The function's name (also the public tool ``name``).
        summary: First line of the docstring — the tool ``description``.
        parameters: One ``_ParameterSpec`` per function parameter, in
            declaration order.
        param_descriptions: Map of parameter name → description text,
            parsed from the docstring's ``Args:`` section. May be empty.
    """

    name: str
    summary: str
    parameters: list[_ParameterSpec]
    param_descriptions: dict[str, str]


# ----------------------------------------------------------------------
# Build result — what the caller needs to form the wire response.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BuildResult:
    """Outcome of a successful build.

    Attributes:
        tool_id: The newly allocated identifier (e.g. ``circle_area_v1``).
        name: The LLM-given function name.
        description: First line of the docstring.
        input_schema: The generated JSON input schema (a plain dict).
        invoke_uri: Relative URI for invoking the tool.
        status: Always ``"PENDING_REVIEW"`` for a freshly built tool.
        catalog_version: The catalog version after this deploy.
        build_duration_ms: Wall-clock build time, for the event log.
    """

    tool_id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    invoke_uri: str
    status: str
    catalog_version: int
    build_duration_ms: int


# Map from a Python builtin type name to its JSON-schema ``type`` string.
# Anything not in this map falls back to an unconstrained schema (``{}``),
# which accepts any JSON value — we'd rather deploy a loosely-typed tool
# than reject a valid one over an exotic annotation.
_PY_TYPE_TO_JSON_TYPE: dict[str, str] = {
    "int": "integer",
    "float": "number",
    "complex": "number",
    "str": "string",
    "bytes": "string",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "frozenset": "array",
    "dict": "object",
}


class Builder:
    """Validates proposed code and persists it as a new tool.

    One instance per ATAF server, sharing the same ``Registry`` and
    ``DeploymentEventLog`` as the rest of the server.

    Thread-safety / concurrency: ``build()`` is NOT internally locked.
    The caller MUST hold the global build lock for the duration of the
    call so that ``tool_id`` allocation and insertion are serialized
    across the whole server (DESIGN.md §5, locked decision).
    """

    def __init__(self, registry: Registry, event_log: DeploymentEventLog) -> None:
        """Construct the builder.

        Args:
            registry: The shared tool registry. Used to allocate a
                ``tool_id`` and persist the validated tool.
            event_log: The shared deployment event log. Receives one
                ``deploy`` (or ``deploy_failed``) event per build.
        """

        self._registry = registry
        self._event_log = event_log

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, *, code: str, intent: str, agent_id: str = "anon") -> BuildResult:
        """Validate, schema-generate, and persist a proposed tool.

        Call this while holding the global build lock. On success the new
        tool is in the registry with status ``PENDING_REVIEW`` and a
        ``deploy`` event has been logged.

        Args:
            code: The complete Python source of the proposed tool.
            intent: The agent's stated intent, stored on the tool row and
                logged for human reviewers.
            agent_id: Best-effort identifier of the proposing agent.
                Defaults to ``"anon"`` in v0.1 (no agent auth yet).

        Returns:
            A ``BuildResult`` describing the deployed tool.

        Raises:
            CodeValidationError: If the code fails any validation rule.
                The caller maps this to a 400 response. A
                ``deploy_failed`` event is logged before the raise.
            DeploymentError: If the code was valid but a server-side step
                (schema, id allocation, or persistence) failed. The
                caller maps this to a 500 response with the stable code
                ``TOOL_NOT_DEPLOYED``. A ``deploy_failed`` event is
                logged before the raise.
        """

        # Start the build clock. We report wall-clock build time in the
        # deploy event so operators can spot pathological builds.
        start = time.perf_counter()

        # Step 1: static validation. On failure we log the rejection
        # (for the audit trail) and re-raise for the caller to turn into
        # a 400. We don't have a tool_id yet, so the event omits it.
        try:
            parsed = self._validate(code)
        except CodeValidationError as err:
            self._event_log.record(
                "deploy_failed",
                reason=err.code,
                message=err.message,
                agent_id=agent_id,
            )
            raise

        # Steps 2-4 run only after validation has passed, so the code is
        # known-good. Any failure from here on is a *server-side*
        # deployment fault (disk write, SQLite insert, id allocation) —
        # not the agent's fault. We wrap them so such failures surface as
        # a single stable TOOL_NOT_DEPLOYED error instead of leaking the
        # raw exception, and so the audit trail still records the attempt.
        try:
            # Step 2: derive the JSON input schema from the parsed
            # signature and docstring. This never fails on its own —
            # unknown annotations degrade to an unconstrained schema
            # rather than raising — but it lives inside the try anyway so
            # the failure contract is uniform across the deploy steps.
            input_schema = self._generate_schema(parsed)
            schema_json = json.dumps(input_schema)

            # Step 3: allocate a unique tool_id. The build lock (held by
            # the caller) guarantees no concurrent allocate/insert race.
            tool_id = self._registry.allocate_tool_id(parsed.name)

            # Record the propose event now that we have a tool_id,
            # mirroring the lifecycle order in DESIGN.md §8.2.1
            # (propose, then deploy).
            self._event_log.record(
                "propose",
                tool_id=tool_id,
                intent=intent,
                agent_id=agent_id,
            )

            # Step 4: persist code + metadata. The registry writes the
            # .py file first, then the SQLite row, rolling back the file
            # if the row insert fails — so a failure here leaves no
            # orphan on disk.
            row = self._registry.insert(
                tool_id=tool_id,
                name=parsed.name,
                description=parsed.summary,
                code=code,
                schema_json=schema_json,
                intent=intent,
            )
        except Exception as err:
            # The proposal was valid but we couldn't deploy it. Log the
            # failed attempt for the audit trail, then re-raise as a
            # stable TOOL_NOT_DEPLOYED for the caller to map to a 500.
            self._event_log.record(
                "deploy_failed",
                reason=DeploymentError.code,
                message=str(err),
                agent_id=agent_id,
            )
            raise DeploymentError(str(err)) from err

        # Step 5: log the successful deploy with its build duration, then
        # hand a fully-formed result back to the caller.
        build_duration_ms = int((time.perf_counter() - start) * 1000)
        self._event_log.record(
            "deploy",
            tool_id=tool_id,
            status=row.status,
            build_duration_ms=build_duration_ms,
        )

        invoke_uri = f"/tools/{tool_id}/invoke"
        return BuildResult(
            tool_id=tool_id,
            name=parsed.name,
            description=parsed.summary,
            input_schema=input_schema,
            invoke_uri=invoke_uri,
            status=row.status,
            catalog_version=self._registry.catalog_version,
            build_duration_ms=build_duration_ms,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, code: str) -> _ParsedTool:
        """Statically validate proposed code and extract its structure.

        Enforces the v0.1 contract from DESIGN.md §3 (Response 2):

          * Parses as valid Python.
          * Contains exactly one top-level function definition and nothing
            else at module level.
          * The function has a Google/NumPy-style docstring.
          * Every parameter is type-annotated; the function has a return
            annotation.
          * No ``*args`` / ``**kwargs`` (we can't schema them in v0.1).
          * Every import (anywhere in the function body) is from the
            Python standard library.

        Args:
            code: The raw proposed source.

        Returns:
            A ``_ParsedTool`` capturing the validated structure.

        Raises:
            CodeValidationError: On the first rule that fails.
        """

        # --- Parse ---
        # A syntax error is the most basic rejection. ast.parse gives us
        # a precise message and line number we can pass back to the LLM.
        try:
            module = ast.parse(code)
        except SyntaxError as err:
            raise CodeValidationError(
                "SYNTAX_ERROR",
                f"code is not valid Python: {err.msg} (line {err.lineno})",
            ) from err

        # --- Exactly one top-level function, nothing else ---
        # We require the module body to be a single function definition.
        # Module-level imports, globals, or extra defs are rejected so the
        # generated tool file is always a clean, single-callable unit.
        if len(module.body) != 1 or not isinstance(
            module.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            raise CodeValidationError(
                "NOT_SINGLE_FUNCTION",
                "code must contain exactly one top-level function definition "
                "and nothing else (imports go inside the function body)",
            )

        func = module.body[0]

        # Async tools aren't supported in v0.1 — the executor runs the
        # function synchronously in a subprocess.
        if isinstance(func, ast.AsyncFunctionDef):
            raise CodeValidationError(
                "ASYNC_NOT_SUPPORTED",
                "async functions are not supported in v0.1",
            )

        # --- Reject *args / **kwargs ---
        # Variadic parameters can't be turned into a fixed JSON schema, so
        # we reject them up front rather than silently dropping them.
        if func.args.vararg is not None or func.args.kwarg is not None:
            raise CodeValidationError(
                "VARIADIC_NOT_SUPPORTED",
                "*args and **kwargs parameters are not supported in v0.1",
            )

        # --- Return annotation required ---
        if func.returns is None:
            raise CodeValidationError(
                "MISSING_RETURN_ANNOTATION",
                f"function {func.name!r} must have a return type annotation",
            )

        # --- Docstring required ---
        # ast.get_docstring returns None if the first statement isn't a
        # bare string literal. A docstring is mandatory: it's the source
        # of the tool description and the parameter descriptions.
        docstring = ast.get_docstring(func)
        if not docstring or not docstring.strip():
            raise CodeValidationError(
                "MISSING_DOCSTRING",
                f"function {func.name!r} must have a docstring",
            )

        # --- Stdlib-only imports ---
        # Walk every Import / ImportFrom anywhere in the function body and
        # confirm each root module is part of the standard library.
        self._check_stdlib_imports(func)

        # --- Extract parameters with their annotations + required-ness ---
        parameters = self._extract_parameters(func)

        # --- Parse the docstring for summary + per-parameter descriptions ---
        summary = self._docstring_summary(docstring)
        param_descriptions = self._docstring_arg_descriptions(docstring)

        # Hand back the structured, validated view.
        return _ParsedTool(
            name=func.name,
            summary=summary,
            parameters=parameters,
            param_descriptions=param_descriptions,
        )

    def _extract_parameters(self, func: ast.FunctionDef) -> list[_ParameterSpec]:
        """Pull the typed parameter list out of a function definition.

        Handles positional-only, normal, and keyword-only parameters.
        Every parameter must be annotated; a missing annotation raises.

        Args:
            func: The validated function definition node.

        Returns:
            One ``_ParameterSpec`` per parameter, in declaration order.

        Raises:
            CodeValidationError: If any parameter lacks a type annotation.
        """

        specs: list[_ParameterSpec] = []

        # Positional parameters are posonlyargs followed by normal args.
        # Defaults align to the *end* of this combined list, so the last
        # len(defaults) positional parameters are the optional ones.
        positional = list(func.args.posonlyargs) + list(func.args.args)
        num_defaults = len(func.args.defaults)
        first_default_index = len(positional) - num_defaults

        for index, arg in enumerate(positional):
            # Every parameter must carry a type annotation.
            if arg.annotation is None:
                raise CodeValidationError(
                    "MISSING_TYPE_ANNOTATION",
                    f"parameter {arg.arg!r} must have a type annotation",
                )
            # A positional parameter is required only if it sits before
            # the first one that has a default value.
            required = index < first_default_index
            specs.append(
                _ParameterSpec(
                    name=arg.arg, annotation=arg.annotation, required=required
                )
            )

        # Keyword-only parameters carry their defaults in kw_defaults,
        # positionally aligned; a None entry means "no default" (required).
        for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
            if arg.annotation is None:
                raise CodeValidationError(
                    "MISSING_TYPE_ANNOTATION",
                    f"parameter {arg.arg!r} must have a type annotation",
                )
            required = default is None
            specs.append(
                _ParameterSpec(
                    name=arg.arg, annotation=arg.annotation, required=required
                )
            )

        return specs

    def _check_stdlib_imports(self, func: ast.FunctionDef) -> None:
        """Reject any import whose root module isn't in the stdlib.

        Walks the whole function subtree so imports nested inside helper
        blocks (``if``, ``try``, etc.) are caught too.

        Args:
            func: The validated function definition node.

        Raises:
            CodeValidationError: On the first non-stdlib import found.
        """

        # sys.stdlib_module_names is the authoritative set of standard
        # library top-level module names for the running interpreter
        # (available since Python 3.10).
        stdlib_modules = sys.stdlib_module_names

        for node in ast.walk(func):
            # `import foo, bar.baz` — check each aliased name's root.
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module not in stdlib_modules:
                        raise CodeValidationError(
                            "NON_STDLIB_IMPORT",
                            f"import of non-stdlib module {root_module!r} is "
                            "not allowed in v0.1 (stdlib only)",
                        )

            # `from foo.bar import baz` — check the module's root. A
            # relative import (level > 0, module is None) is meaningless
            # for a standalone tool file, so reject it too.
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    raise CodeValidationError(
                        "NON_STDLIB_IMPORT",
                        "relative imports are not allowed in tool code",
                    )
                root_module = (node.module or "").split(".")[0]
                if root_module not in stdlib_modules:
                    raise CodeValidationError(
                        "NON_STDLIB_IMPORT",
                        f"import from non-stdlib module {root_module!r} is "
                        "not allowed in v0.1 (stdlib only)",
                    )

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------

    def _generate_schema(self, parsed: _ParsedTool) -> dict[str, Any]:
        """Build a JSON input schema from a parsed tool.

        The shape mirrors Anthropic's tool-use ``input_schema`` field and
        OpenAPI parameter schemas (DESIGN.md §4)::

            {
              "type": "object",
              "properties": {
                "radius": {"type": "number", "description": "..."}
              },
              "required": ["radius"]
            }

        Args:
            parsed: The validated tool structure.

        Returns:
            A JSON-serializable dict describing the tool's input.
        """

        properties: dict[str, Any] = {}
        required: list[str] = []

        for spec in parsed.parameters:
            # Map the Python annotation to a JSON-schema fragment.
            property_schema = self._annotation_to_schema(spec.annotation)

            # Attach the docstring description for this parameter, if the
            # author wrote one. Purely additive — helps the LLM choose args.
            description = parsed.param_descriptions.get(spec.name)
            if description:
                property_schema["description"] = description

            properties[spec.name] = property_schema

            # Parameters without a default are required.
            if spec.required:
                required.append(spec.name)

        # Always emit "type": "object". Include "required" only when at
        # least one parameter is mandatory, matching common JSON-schema
        # convention (an empty "required" array is noise).
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema

    def _annotation_to_schema(self, annotation: ast.expr) -> dict[str, Any]:
        """Map a single AST type annotation to a JSON-schema fragment.

        Supported shapes:
          * Bare builtins: ``int``, ``float``, ``str``, ``bool``, ``dict``,
            ``list``, etc. (via ``_PY_TYPE_TO_JSON_TYPE``).
          * ``list[T]`` / ``List[T]`` → ``{"type": "array", "items": ...}``.
          * ``dict[K, V]`` / ``Dict`` → ``{"type": "object"}``.
          * ``Optional[T]`` and ``T | None`` → the schema for ``T``.

        Anything unrecognized degrades to ``{}`` (accept any JSON value)
        rather than raising — we never reject a valid tool over an exotic
        annotation we simply don't model yet in v0.1.

        Args:
            annotation: The AST annotation node for one parameter.

        Returns:
            A JSON-schema fragment dict for that parameter.
        """

        # Case 1: a bare name like `float` or `list`.
        if isinstance(annotation, ast.Name):
            json_type = _PY_TYPE_TO_JSON_TYPE.get(annotation.id)
            if json_type is None:
                # Unknown bare type (e.g. a custom class) — accept anything.
                return {}
            return {"type": json_type}

        # Case 2: a subscripted generic like `list[int]`, `dict[str, int]`,
        # or `Optional[int]`.
        if isinstance(annotation, ast.Subscript):
            return self._subscript_to_schema(annotation)

        # Case 3: a PEP 604 union like `int | None`. We unwrap a single
        # non-None member (the common "optional" pattern) and schema that;
        # broader unions fall back to "accept anything".
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return self._union_to_schema(annotation)

        # Case 4: a literal None annotation (rare for a parameter) or
        # anything else we don't model — accept any value.
        return {}

    def _subscript_to_schema(self, node: ast.Subscript) -> dict[str, Any]:
        """Map a subscripted generic annotation to a JSON-schema fragment.

        Args:
            node: An ``ast.Subscript`` such as ``list[int]`` or
                ``Optional[str]``.

        Returns:
            A JSON-schema fragment dict.
        """

        # The base of the subscript tells us the container kind. It may be
        # a bare name (`list`) or a dotted attribute (`typing.List`); we
        # only need the final segment.
        base_name = self._base_name(node.value)

        # Optional[T] is just T made not-required. We model the value type
        # itself; required-ness is decided separately from the default.
        if base_name in ("Optional",):
            return self._annotation_to_schema(node.slice)

        # Union[...] — unwrap a single non-None member if possible.
        if base_name in ("Union",):
            return self._union_members_to_schema(self._tuple_elements(node.slice))

        # list[T] / List[T] / set[T] / tuple[...] → JSON array. Carry the
        # element schema through as "items" when there's a single element.
        if base_name in ("list", "List", "set", "Set", "frozenset", "FrozenSet"):
            element_schema = self._annotation_to_schema(node.slice)
            array_schema: dict[str, Any] = {"type": "array"}
            if element_schema:
                array_schema["items"] = element_schema
            return array_schema

        # tuple[...] → array; we don't model per-position item types in v0.1.
        if base_name in ("tuple", "Tuple"):
            return {"type": "array"}

        # dict[K, V] / Dict[...] → JSON object.
        if base_name in ("dict", "Dict", "Mapping"):
            return {"type": "object"}

        # Unknown generic — accept anything.
        return {}

    def _union_to_schema(self, node: ast.BinOp) -> dict[str, Any]:
        """Map a PEP 604 ``A | B`` union annotation to a schema fragment.

        Args:
            node: An ``ast.BinOp`` whose operator is ``|``.

        Returns:
            The schema for the single non-None member if the union is
            ``T | None``; otherwise an unconstrained ``{}``.
        """

        # Flatten the (possibly nested) union into its member nodes, drop
        # any explicit None, and reuse the shared member-selection logic.
        members = self._flatten_bitor(node)
        return self._union_members_to_schema(members)

    def _union_members_to_schema(self, members: list[ast.expr]) -> dict[str, Any]:
        """Pick a schema for a union given its member annotation nodes.

        The common, modellable case is ``T | None`` (or ``Optional[T]``):
        exactly one non-None member, which we schema directly. Any wider
        union degrades to an unconstrained ``{}``.

        Args:
            members: The union's member annotation nodes.

        Returns:
            A JSON-schema fragment dict.
        """

        # Drop explicit None members — they only signal optionality, which
        # is handled by the required list, not the value schema.
        non_none = [m for m in members if not self._is_none_literal(m)]

        # Exactly one real type left → schema it. Otherwise we can't pick
        # a single JSON type cleanly, so accept anything.
        if len(non_none) == 1:
            return self._annotation_to_schema(non_none[0])
        return {}

    # ------------------------------------------------------------------
    # Small AST helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_name(node: ast.expr) -> str:
        """Return the final identifier of a name or dotted-attribute node.

        ``list`` → ``"list"``; ``typing.List`` → ``"List"``. Used to
        recognize generics regardless of how they were imported.

        Args:
            node: A ``ast.Name`` or ``ast.Attribute`` node.

        Returns:
            The trailing identifier, or ``""`` for anything else.
        """

        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _flatten_bitor(self, node: ast.expr) -> list[ast.expr]:
        """Flatten a nested ``A | B | C`` BinOp tree into a member list.

        Args:
            node: A union annotation node (BinOp with ``|``) or a leaf.

        Returns:
            The flattened list of member annotation nodes.
        """

        # Recurse into both sides of every BitOr; a non-union node is a
        # single leaf member.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._flatten_bitor(node.left) + self._flatten_bitor(node.right)
        return [node]

    @staticmethod
    def _tuple_elements(node: ast.expr) -> list[ast.expr]:
        """Return the element nodes of an ``ast.Tuple`` subscript slice.

        ``Union[int, str]`` parses its slice as a tuple ``(int, str)``;
        a single-element slice like ``Optional[int]`` is just the bare
        node. This normalizes both into a flat list.

        Args:
            node: The ``.slice`` of a subscript.

        Returns:
            The element nodes.
        """

        if isinstance(node, ast.Tuple):
            return list(node.elts)
        return [node]

    @staticmethod
    def _is_none_literal(node: ast.expr) -> bool:
        """True if an annotation node is the literal ``None``.

        Recognizes both ``None`` as a constant and the legacy
        ``type(None)`` spelling is out of scope — only the constant form
        appears in real annotations.

        Args:
            node: An annotation node.

        Returns:
            Whether the node denotes ``None``.
        """

        return isinstance(node, ast.Constant) and node.value is None

    # ------------------------------------------------------------------
    # Docstring parsing (Google style)
    # ------------------------------------------------------------------

    @staticmethod
    def _docstring_summary(docstring: str) -> str:
        """Return the first non-empty line of a docstring as the summary.

        This becomes the tool's ``description`` in the catalog.

        Args:
            docstring: The cleaned docstring (``ast.get_docstring`` output).

        Returns:
            The first non-empty, stripped line. Empty string only if the
            docstring is entirely whitespace (already rejected upstream).
        """

        for line in docstring.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    @staticmethod
    def _docstring_arg_descriptions(docstring: str) -> dict[str, str]:
        """Parse a Google-style ``Args:`` section into name → description.

        Recognizes the block that starts at an ``Args:`` (or
        ``Arguments:``) header and runs until the next section header
        (``Returns:``, ``Raises:``, etc.) or the end of the docstring.
        Each argument line looks like ``name: description`` or
        ``name (type): description``; continuation lines indented further
        are appended to the current argument's description.

        This is a small hand-rolled parser (plan.txt open question #2 —
        no ``docstring-parser`` dependency in v0.1). It is intentionally
        forgiving: anything it can't parse is simply skipped, never raised.

        Args:
            docstring: The cleaned docstring.

        Returns:
            Map of parameter name → description text. Empty if there is no
            recognizable ``Args:`` section.
        """

        # The set of section headers that terminate the Args block. Kept
        # lowercase for case-insensitive matching.
        section_headers = {
            "returns:", "return:", "raises:", "yields:", "yield:",
            "examples:", "example:", "note:", "notes:", "attributes:",
            "args:", "arguments:", "parameters:",
        }

        lines = docstring.splitlines()

        # First, find where the Args section starts.
        in_args = False
        descriptions: dict[str, str] = {}
        current_name: str | None = None

        for line in lines:
            stripped = line.strip()
            lowered = stripped.lower()

            # Toggle into the Args section on its header line.
            if lowered in ("args:", "arguments:", "parameters:"):
                in_args = True
                current_name = None
                continue

            # If we're in the Args section and hit another section header,
            # the Args block is over — stop scanning.
            if in_args and lowered in section_headers:
                break

            if not in_args:
                continue

            # Blank line inside Args — just skip it, don't end the section.
            if not stripped:
                continue

            # A "name: description" line starts a new argument. We split on
            # the first colon only, so descriptions may contain colons.
            if ":" in stripped:
                name_part, _, desc_part = stripped.partition(":")
                # Strip an optional "(type)" suffix from the name part, e.g.
                # "radius (float)" → "radius".
                name = name_part.split("(")[0].strip()
                # Only accept identifier-like names; anything else is treated
                # as a continuation line for the current argument.
                if name.isidentifier():
                    descriptions[name] = desc_part.strip()
                    current_name = name
                    continue

            # Otherwise this is a continuation of the current argument's
            # description — append it with a single separating space.
            if current_name is not None:
                existing = descriptions.get(current_name, "")
                joined = f"{existing} {stripped}".strip()
                descriptions[current_name] = joined

        return descriptions
