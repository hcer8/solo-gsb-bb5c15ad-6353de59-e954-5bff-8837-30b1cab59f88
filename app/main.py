"""FastAPI application: three-way JSON merge for configuration review."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .checks import evaluate_check_rules, parse_rules, render_scope
from .config import settings
from .merger import Merger, parse_pointer
from .middleware import RequestSizeLimitMiddleware
from .models import (
    Conflict,
    MergeRequest,
    MergeResponse,
    ProvenanceEntry,
    Violation,
)
from .patch import build_json_patch

app = FastAPI(
    title="Three-way JSON Merge API",
    description=(
        "Merges two JSON revisions of a configuration against their common "
        "base. Changes on one side are accepted automatically; divergent "
        "changes and delete-versus-modify cases keep the base value and are "
        "reported as conflicts located by JSON Pointer (RFC 6901). "
        "'protectedPaths' marks pointers (a '*' token matches one segment) "
        "that no change may write automatically: any left/right change "
        "there, including one-sided or identical changes, keeps the base "
        "and is reported with kind 'protected_change'. Reported conflicts "
        "may be adjudicated per request via 'conflictResolutions', which "
        "maps a conflict pointer to the winning side (base, left or "
        "right); explicit adjudication also overrides protected paths. "
        "'redactPaths' masks sensitive values in the reported conflicts "
        "only: base/left/right values at (or below) a listed pointer are "
        "replaced with '[REDACTED]', while 'merged', conflict detection "
        "and adjudication keep using the original values. With "
        "'includeProvenance' the response additionally carries a "
        "'provenance' array recording, for every actual changed leaf or "
        "identity-array item, the side its final value came from "
        "(base, left, right, merged), whether a conflict adjudication "
        "produced it, and whether it was deleted; records are sorted by "
        "final JSON Pointer (deleted items keep their original pointer). "
        "With 'includePatch' the response additionally carries a 'patch' "
        "array of RFC 6902 JSON Patch operations that turn the request "
        "'base' into the response 'merged' when applied in order: object "
        "differences are emitted recursively, arrays are replaced "
        "wholesale, and reference tokens follow RFC 6901 escaping. "
        "Without redaction the response conflicts always keep the original "
        "three-way values. With 'checkRules' the response additionally "
        "carries a 'violations' array: each rule constrains the direct "
        "fields of the object scopes its RFC 6901 'scope' pointer expands "
        "to (a '*' token matches one single level of object keys) with "
        "mode 'all_or_none' (either every listed field is present or none "
        "is) or 'at_most_one' (at most one listed field may be present; "
        "an existing key counts, null included). Rules are evaluated "
        "against 'base' and against the final, adjudicated 'merged'; only "
        "violations newly present in merged are reported - a violation "
        "already present in base that persists, or a scope deleted from "
        "merged, is not reported. Entries carry the rule index, the "
        "concrete scope pointer and the listed fields actually present, "
        "sorted by rule index then scope pointer."
    ),
    version="1.6.0",
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_size=settings.max_request_size_bytes,
)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/merge",
    response_model=MergeResponse,
    tags=["merge"],
    summary="Three-way merge of JSON documents",
)
async def merge(request: MergeRequest):
    # Validate pointers up front so malformed configuration produces a clear
    # 4xx response instead of an opaque error.
    for pointer in request.array_identity:
        try:
            parse_pointer(pointer)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid arrayIdentity pointer {pointer!r}: {exc}",
            ) from exc

    # Protected pointers must be well-formed and unique (by decoded
    # reference tokens, so escaped spellings of the same pointer collide).
    protected_seen: set[tuple[str, ...]] = set()
    for pointer in request.protected_paths:
        try:
            tokens = parse_pointer(pointer)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid protectedPaths pointer {pointer!r}: {exc}",
            ) from exc
        if tokens in protected_seen:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate protectedPaths entry for pointer {pointer!r}",
            )
        protected_seen.add(tokens)

    # Redact pointers must be well-formed and unique (by decoded
    # reference tokens, so escaped spellings of the same pointer collide).
    redact_seen: set[tuple[str, ...]] = set()
    for pointer in request.redact_paths:
        try:
            tokens = parse_pointer(pointer)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid redactPaths pointer {pointer!r}: {exc}",
            ) from exc
        if tokens in redact_seen:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate redactPaths entry for pointer {pointer!r}",
            )
        redact_seen.add(tokens)

    # Check rules must be well-formed: a valid RFC 6901 scope pointer and
    # at least two distinct, non-empty direct field names. Like the other
    # toggles, an explicitly supplied list enables the "violations" key
    # (even when empty); omission leaves the legacy envelope untouched.
    rules = request.check_rules
    rules_enabled = "check_rules" in request.model_fields_set and rules is not None
    if rules is not None:
        for index, rule in enumerate(rules):
            try:
                parse_pointer(rule.scope)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"invalid checkRules[{index}] scope pointer "
                        f"{rule.scope!r}: {exc}"
                    ),
                ) from exc
            if len(rule.fields) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"checkRules[{index}] must list at least two fields"
                    ),
                )
            if len(set(rule.fields)) != len(rule.fields):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"checkRules[{index}] fields must not contain "
                        "duplicates"
                    ),
                )
            if any(not field for field in rule.fields):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"checkRules[{index}] field names must be "
                        "non-empty strings"
                    ),
                )

    try:
        merger = Merger(
            request.array_identity, request.protected_paths, request.redact_paths
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate resolution pointers up front: they must be well-formed and
    # unique (by decoded reference tokens, so escaped spellings of the same
    # pointer collide). Whether each pointer denotes a real conflict can
    # only be decided by the merger, which raises ValueError -> 400.
    resolutions: dict[str, str] = {}
    seen: set[tuple[str, ...]] = set()
    for entry in request.conflict_resolutions:
        try:
            tokens = parse_pointer(entry.pointer)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"invalid conflictResolutions pointer {entry.pointer!r}: {exc}",
            ) from exc
        if tokens in seen:
            raise HTTPException(
                status_code=400,
                detail=(
                    "duplicate conflictResolutions entry for pointer "
                    f"{entry.pointer!r}"
                ),
            )
        seen.add(tokens)
        resolutions[entry.pointer] = entry.resolution

    try:
        if request.include_provenance:
            merged, raw_conflicts, raw_provenance = merger.merge_with_provenance(
                request.base, request.left, request.right, resolutions
            )
        else:
            merged, raw_conflicts = merger.merge(
                request.base, request.left, request.right, resolutions
            )
            raw_provenance = []
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    conflicts = [
        Conflict(
            pointer=conflict.path,
            kind=conflict.kind,
            base=conflict.base,
            left=conflict.left,
            right=conflict.right,
        )
        for conflict in raw_conflicts
    ]
    provenance = [
        ProvenanceEntry(
            pointer=record.path,
            source=record.source,
            adjudicated=record.adjudicated,
            deleted=record.deleted,
        )
        for record in raw_provenance
    ] if request.include_provenance else None

    # Serialize directly: with the optional toggles off the response must be
    # byte-for-byte the old two-key envelope (no "provenance": null /
    # "patch": null / "violations": null), while an enabled request always
    # includes the matching key, even when its array is empty.
    body: dict = {
        "merged": merged,
        "conflicts": [c.model_dump() for c in conflicts],
    }
    if request.include_provenance:
        body["provenance"] = [entry.model_dump() for entry in provenance]
    if request.include_patch:
        body["patch"] = build_json_patch(request.base, merged)
    if rules_enabled:
        # Pure post-processing: the merge above (protection, adjudication
        # included) is untouched; only new violations of the final merged
        # document are reported.
        violations = evaluate_check_rules(request.base, merged, parse_rules(rules))
        body["violations"] = [
            Violation(
                rule=v.rule,
                scope=render_scope(v.scope),
                fields=v.fields,
            ).model_dump()
            for v in violations
        ]
    return JSONResponse(body)
