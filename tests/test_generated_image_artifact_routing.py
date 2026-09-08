"""Offline contrasts for the HTML-IMAGE-SMOKE-01 producer-routing defect.

The saved prompt is embedded so this regression never reads production state.
Graph assertions follow the real IR promotion boundary; generated file truth is
checked separately through deterministic, temporary-root backend execution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ollmo_g.request_meta import extract_request_meta
from ollmo_g.request_phase_graph import build_request_phase_graph
from ollmo_server.response_semantics_runtime import ResponseSemanticsRuntimeOwner
from ollmo_services.responses import build_canonical_response_artifacts
from tests.fake_backends import FakeBackendHarness
from tests.fake_backends.fixtures import write_text


SAVED_SMOKE_PROMPT = '''Generate exactly one new image: a simple illustration of a small orange sailboat
on a calm blue lake, without lettering. Save the generated image locally as
html-image-01.png.

Create a small HTML page named html-image-01.html with the heading "Ein Tag am See"
and the short text "Ein kleines Segelboot auf einem ruhigen blauen See."
The page must display exactly the newly generated local image, once, through a
working relative file path to its actual saved filename.

Create a separate stylesheet named html-image-01.css. Link exactly this stylesheet
from the HTML through a working relative path. Use it for a light blue page
background, a dark blue heading and a responsive image. Keep all CSS in this file;
do not use embedded CSS or inline style attributes.

Return exactly these three files: the HTML page, its CSS file, and the one newly
generated PNG image. The three files must work together locally without external
resources. Do not create JavaScript, audio, additional pages, additional images,
or a bundle. Do not use placeholders, external image URLs, data URLs, Base64 images,
or an SVG/HTML drawing as a substitute for actual image generation.'''

WEB_PROMPT = (
    "Generate one PNG image of an orange sailboat on a blue lake. "
    "Create index.html and a separate styles.css. "
    "The HTML must display the generated local image through a relative path. "
    "Return exactly these three files."
)


def _graph(prompt):
    return build_request_phase_graph(
        prompt,
        request_payload={"prompt": prompt, "ghost_route": True},
        route_payload={"capability": "chat", "route_source": "ghost_carried"},
    )


def _image_items(items):
    return [item for item in items if item.get("capability") == "image_generation"]


@pytest.fixture
def semantics():
    return ResponseSemanticsRuntimeOwner(
        hooks={
            "build_canonical_response_artifacts": build_canonical_response_artifacts,
            "normalize_capability_list": lambda values: list(dict.fromkeys(values or [])),
            "extract_request_meta": extract_request_meta,
            "extract_responses_prompt": lambda payload: str((payload or {}).get("prompt") or ""),
            "resolve_semantic_review_artifact_path": lambda path: (
                Path(path).resolve() if Path(path).is_file() else None
            ),
            "load_running_instances": lambda: [],
            "merge_instances_with_runtime_status": lambda instances, **kwargs: instances,
        }
    )


@pytest.mark.parametrize(
    "prompt,image_count,text_extensions",
    [
        ("Generate one PNG image.", 1, []),
        ("Create an SVG illustration.", 0, ["svg"]),
        ("Generate PNG; do not use SVG.", 1, []),
        ("Do not generate PNG and create SVG.", 0, []),
        ('HTML references image.png through <img src="image.png">.', 0, []),
        (WEB_PROMPT, 1, ["css", "html"]),
        (WEB_PROMPT + " No SVG, no Base64, no data URL, no placeholder.", 1, ["css", "html"]),
        (SAVED_SMOKE_PROMPT, 1, ["css", "html"]),
        ("Create an SVG illustration and a PNG image.", 1, ["svg"]),
        ("Create one PNG image and an SVG illustration.", 1, ["svg"]),
        ("Create an SVG and a PNG illustration.", 1, ["svg"]),
        ("Create a PNG and an SVG illustration.", 1, ["svg"]),
    ],
    ids=[
        "png", "explicit-svg", "png-not-svg", "coordinated-prohibition", "reference-only", "three-files",
        "negative-formats", "saved-live-prompt", "svg-illustration-and-png-image",
        "png-image-and-svg-illustration", "svg-and-png", "png-and-svg",
    ],
)
def test_requested_formats_reach_only_their_promoted_producers(
    semantics, prompt, image_count, text_extensions
):
    graph = _graph(prompt)
    ir = graph["request_ir"]
    obligations = ir["output_obligations"]
    image_obligations = _image_items(obligations)
    assert len(image_obligations) == image_count
    assert sorted(
        item["text_artifact_extension"] for item in obligations
        if item.get("role") == "text_artifact_output"
    ) == text_extensions

    # Output candidates must have a runtime-reviewed contract; workload-task
    # orientation candidates are intentionally a separate, advisory family.
    candidates = [
        item for item in ir["candidate_graph"]["candidates"]
        if item.get("candidate_type") == "output"
    ]
    image_candidates = _image_items(candidates)
    assert len(image_candidates) == image_count
    decisions = {item["candidate_id"]: item for item in ir["promotion_review"]["decisions"]}
    for candidate in image_candidates:
        assert candidate["status"] == "promoted"
        decision = decisions[candidate["candidate_id"]]
        assert decision["decision"] == "promoted"
        assert decision["authority"] == "runtime_review"
        assert decision["execution_policy"] == "executable_obligation"
        assert decision["contract_ref"] == candidate["obligation_id"]

    assert len(_image_items(graph["intent_obligations"])) == image_count
    image_phases = _image_items(graph["phases"])
    image_branches = _image_items(graph.get("downstream_branches") or [])
    assert len(image_phases) == len(image_branches) == image_count
    for branch in image_branches:
        assert branch.get("contract_state") != "reserved"
        assert branch["output_type"] == "image"
        assert branch["output_contract"]["fulfillment_policy"] == "runtime_artifact_or_branch_state"
        assert branch["obligation_id"] in {item["obligation_id"] for item in image_obligations}
    assert sorted(
        item["text_artifact_extension"] for item in graph.get("downstream_branches") or []
        if item.get("role") == "text_artifact_output"
    ) == text_extensions

    pending = semantics.extract_pending_deferred_branches(
        route_payload={"route_runtime": {"request_phase_graph": graph}}
    )
    assert len(_image_items(pending)) == image_count

    if image_count and text_extensions == ["css", "html"]:
        # Exactly three requested file artifacts, regardless of preparation or
        # semantic-review obligations which do not create user files.
        file_obligations = [
            item for item in obligations
            if item.get("role") == "text_artifact_output" or item.get("output_type") == "image"
        ]
        assert len(file_obligations) == 3
        image_ids = {item["phase_id"] for item in image_phases}
        for branch in graph["downstream_branches"]:
            if branch.get("role") == "text_artifact_output":
                assert image_ids <= set(branch["depends_on"])
                assert branch["dependency_contract"] == "local_visual_asset_binding"


@pytest.mark.parametrize(
    "prompt",
    [
        "Generate PNG documentation.",
        "Generate one PNG example filename.",
        "Generate PNG later; create the HTML now.",
        "Create one HTML file. Do not use placeholders, external image URLs, data URLs, Base64 images,\n"
        "or an SVG/HTML drawing as a substitute for actual image generation.",
    ],
    ids=["raster-documentation", "raster-example-filename", "raster-deferred", "html-with-exclusions-only"],
)
def test_raster_mentions_without_current_production_have_no_promoted_image(semantics, prompt):
    graph = _graph(prompt)
    ir = graph["request_ir"]
    assert not _image_items(ir["output_obligations"])
    assert not _image_items(graph["intent_obligations"])
    image_candidates = _image_items(ir["candidate_graph"]["candidates"])
    image_candidate_ids = {item["candidate_id"] for item in image_candidates}
    assert all(item.get("execution_policy") != "executable_obligation" for item in image_candidates)
    assert all(
        item["decision"] != "promoted"
        for item in ir["promotion_review"]["decisions"]
        if item["candidate_id"] in image_candidate_ids
    )
    assert not [
        branch for branch in _image_items(graph.get("downstream_branches") or [])
        if branch.get("contract_state") != "reserved"
    ]
    pending = semantics.extract_pending_deferred_branches(
        route_payload={"route_runtime": {"request_phase_graph": graph}}
    )
    assert not _image_items(pending)


def test_saved_prompt_identity():
    import hashlib

    assert hashlib.sha256(SAVED_SMOKE_PROMPT.encode()).hexdigest() == (
        "b5d994a73ed0ec069bff10620ddeaa1fcbc698db1e422295643361e37069c654"
    )


@pytest.mark.parametrize("prompt", [
    SAVED_SMOKE_PROMPT,
    "Create a separate stylesheet named styles.css. Use it for a blue page background.",
    "Create index.html with a short poem. Turn it into an image.",
])
def test_graph_prepared_named_source_survives_sentence_boundary(semantics, prompt):
    graph = _graph(prompt)
    payload = {"id": "offline_named_source", "output_text": "Image prompt: An orange sailboat."}
    updated = semantics.truth_gate_response_output_claims(
        payload, request_payload={"prompt": prompt},
        route_payload={"capability": "chat", "route_runtime": {"request_phase_graph": graph}},
    )
    assert updated["output_text"] == payload["output_text"]
    assert not updated.get("runtime", {}).get("truth_guard")


@pytest.mark.parametrize("prompt", [
    "Use it for a blue page background.",
    "Do not create styles.css. Use it for a blue page background.",
    'Example: "Create styles.css". Use it for a blue page background.',
    "Create styles.css from the missing source. Use it for a blue page background.",
    "Create styles.css. Select the uploaded document. Turn it into an image.",
])
def test_named_source_does_not_ground_external_or_unrequested_input(semantics, prompt):
    updated = semantics.truth_gate_response_output_claims(
        {"output_text": "Prepared content."}, request_payload={"prompt": prompt},
        route_payload={"capability": "chat", "route_runtime": {"request_phase_graph": _graph(prompt)}},
    )
    assert updated["runtime"]["truth_guard"]["status"] == "clarification_required"


@pytest.mark.parametrize("fault", ["absent", "reserved", "wrong_name", "not_required", "disconnected"])
def test_named_source_requires_its_exact_owed_graph_target(semantics, fault):
    prompt = "Create a separate stylesheet named styles.css. Use it for a blue page background."
    graph = _graph(prompt)
    phase = next(p for p in graph["phases"] if p.get("role") == "text_artifact_output")
    if fault == "absent":
        graph["phases"].remove(phase)
    elif fault == "reserved":
        phase["contract_state"] = "reserved"
    elif fault == "wrong_name":
        phase["artifact_request"]["source_name"] = "unrelated"
    elif fault == "not_required":
        phase["output_contract"]["required"] = False
    else:
        phase["depends_on"] = []
    updated = semantics.truth_gate_response_output_claims(
        {"output_text": "Prepared content."}, request_payload={"prompt": prompt},
        route_payload={"capability": "chat", "route_runtime": {"request_phase_graph": graph}},
    )
    assert updated["runtime"]["truth_guard"]["status"] == "clarification_required"


@pytest.mark.parametrize("status", ["pending", "planned", "blocked", "failed", "active", "deferred"])
@pytest.mark.parametrize("capability,output_type", [("image_generation", "image"), ("text_to_speech", "audio")])
def test_terminal_materialization_retains_open_media_obligation(status, capability, output_type):
    from ollmo_server.late_fill_runtime import LateFillRuntimeOwner

    check = {"branch_id": "branch-image_generation-1", "phase_id": "phase-2",
             "role": "final_output", "capability": capability, "output_type": output_type,
             "status": status, "evidence": "pending_graph_branch", "obligation_id": "obligation-phase-2"}
    payload = {"runtime": {"graph_closure_review": {"checks": [check]}}}
    assert LateFillRuntimeOwner._terminal_materialization_contract_open_checks(payload) == [check]


@pytest.mark.parametrize("status", ["fulfilled", "waived", "superseded", "reserved", "candidate"])
def test_terminal_materialization_does_not_reopen_closed_or_optional_media(status):
    from ollmo_server.late_fill_runtime import LateFillRuntimeOwner

    check = {"capability": "image_generation", "output_type": "image", "role": "final_output",
             "obligation_id": "obligation-phase-2",
             "status": status, "evidence": "pending_graph_branch"}
    assert not LateFillRuntimeOwner._terminal_materialization_contract_open_checks(
        {"runtime": {"graph_closure_review": {"checks": [check]}}}
    )


def test_html_reference_cannot_fulfill_a_requested_png(semantics, tmp_path):
    graph = _graph(WEB_PROMPT)
    html = tmp_path / "index.html"
    css = tmp_path / "styles.css"
    html.write_text('<!doctype html><link rel="stylesheet" href="styles.css"><img src="image.png">')
    css.write_text("body { background: lightblue; } img { max-width: 100%; }")
    artifacts = [
        {"type": "text", "path": str(path), "name": path.stem, "mime_type": mime, "content": path.read_text()}
        for path, mime in [(html, "text/html"), (css, "text/css")]
    ]
    payload = {
        "id": "offline_missing_png",
        "output_text": "The image is generated and linked.",
        "runtime": {"request_phase_graph": graph},
        "artifacts": artifacts,
    }
    review = semantics.build_graph_closure_review(
        payload["output_text"], request_payload={"prompt": WEB_PROMPT, "ghost_route": True},
        artifact_payload=payload,
    )
    image_checks = _image_items(review["checks"])
    assert image_checks
    assert all(item["status"] not in {"completed", "fulfilled"} for item in image_checks)
    assert not (tmp_path / "image.png").exists()
    # Exercise final closure too: text files and a successful CSS binding must
    # not erase the missing promoted media obligation.
    import ollmo_webserver

    with _GeneratedWebHarness() as harness:
        finalized, status = ollmo_webserver._LATE_FILL_RUNTIME.finalize_terminal_materialization_contract(
            payload, request_payload={"prompt": WEB_PROMPT, "ghost_route": True},
            route_payload={"capability": "chat", "route_runtime": {"request_phase_graph": graph}},
            artifact_gap={}, terminal_status="completed",
        )
        assert status != "completed"
        assert finalized["late_fill"]["final_materialization_contract_status"] == "unmet"
        assert _image_items(finalized["late_fill"]["materialization_contract_open_checks"])
        assert harness.calls["image_generation"] == 0


class _GeneratedWebHarness(FakeBackendHarness):
    """Exercise graph, branch-contract, result and final-materialization owners.

    The inherited harness supplies fake instances, routing, target selection,
    request preparation and provider I/O, plus temporary storage and disabled
    background scheduler/log hooks. It does not exercise live Ghost routing.
    """

    def _build_instances(self):
        instances = super()._build_instances()
        for instance in instances.values():
            # These endpoints are in-process fakes. Supply simulated positive
            # liveness to the unchanged selection gate, without opening ports.
            instance["runtime_status"].update(process_alive=True, port_listening=True)
        return instances

    def _resolve_ghost_auto_route(self, data, *args, **kwargs):
        route, error = super()._resolve_ghost_auto_route(data, *args, **kwargs)
        if data.get("ghost_route") and not data.get("capability"):
            graph = _graph(data["prompt"])
            route["route_runtime"]["request_phase_graph"] = graph
        return route, error

    def _execute_chat_backend_request(self, **kwargs):
        super()._execute_chat_backend_request(**kwargs)
        return "Image prompt: A simple illustration of an orange sailboat on a calm blue lake."

    def _persist_generated_text_artifact_if_requested(self, *args, **kwargs):
        # Preparation text is not any of the three requested files. Text files
        # are written only by the later fake chat materializer.
        return {}

    def _invoke_internal_api_json_route(self, path=None, *, payload=None, upload=None):
        data = dict(payload or {})
        if self._capability_from_payload(data) != "chat":
            result, status = super()._invoke_internal_api_json_route(path, payload=data, upload=upload)
            if self._capability_from_payload(data) == "image_generation":
                # The legacy tiny-PNG fixture has an invalid IDAT CRC. This
                # provider-only fixture emits a complete valid RGB PNG.
                import struct
                import zlib

                def chunk(kind, body):
                    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

                header = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)
                pixels = b"\x00\xff\x80\x00\x00\x80\xff"
                Path(result["saved_image_path"]).write_bytes(
                    b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                    + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
                )
            return result, status
        image = self.images_dir / "generated.png"
        assert image.is_file(), "HTML/CSS must execute after its accepted image producer"
        self.calls["chat_materialization"] += 1
        html = self.web_dir / "index.html"
        css = self.web_dir / "styles.css"
        import os

        relative_image = os.path.relpath(image, self.web_dir)
        targets = data.get("text_artifact_requests") or [data.get("artifact_request") or {}]
        extensions = {item.get("extension") for item in targets}
        assert extensions <= {"html", "css"} and extensions
        files = []
        if "html" in extensions:
            write_text(html, '<!doctype html><html><head><link rel="stylesheet" href="styles.css">'
                       '</head><body><h1>Ein Tag am See</h1><p>Ein kleines Segelboot auf einem '
                       f'ruhigen blauen See.</p><img src="{relative_image}" alt="Sailboat"></body></html>')
            files.append((html, "text/html"))
        if "css" in extensions:
            write_text(css, 'body { background-color: lightblue; } h1 { color: darkblue; } '
                       'img { max-width: 100%; height: auto; }')
            files.append((css, "text/css"))
        result = {
            "mode": "chat",
            "content": "Requested local web files are saved.",
            "saved_text_path": str(files[0][0]),
            "saved_text_artifacts": [
                {"path": str(file), "mime_type": mime,
                 "text_artifact_request": {"source_name": file.stem, "extension": file.suffix[1:]}}
                for file, mime in files
            ],
        }
        self.call_records.append({"capability": "chat", "payload": data, "result": result})
        return result, 200


def _record_executed_branch(runtime, response, execution):
    """Supply callback evidence only after the real executor returned bytes."""
    result = execution["infer_result"]
    contract = execution["execution_contract"]
    record = {
        **result,
        **{key: contract[key] for key in ("branch_id", "phase_id", "capability")},
        "execution_contract": contract,
        "status": "completed",
    }
    record = runtime.attach_late_fill_result_artifact_identity(
        record, response, result, capability=contract["capability"],
    )
    state = response.get("late_fill") or {}
    state = {
        **state, "status": "running",
        "fill_results": [*state.get("fill_results", []), record],
        "completed_branches": [*state.get("completed_branches", []), record],
    }
    return runtime.merge_late_fill_result_into_response_payload(response, result, state)


@pytest.mark.parametrize("prompt", [WEB_PROMPT, SAVED_SMOKE_PROMPT], ids=["compact", "saved-live-prompt"])
def test_generated_png_provider_executes_before_web_materialization(semantics, prompt):
    """Bounded integration through real branch and final-contract owners.

    The asynchronous scheduler/frame persistence is outside this regression;
    each executed result is supplied as its ordinary callback evidence.
    """
    import hashlib
    import ollmo_webserver
    import struct
    import zlib

    with _GeneratedWebHarness() as harness:
        graph = _graph(prompt)
        response = {"id": "offline_generated_web", "output_text": "Image prompt: An orange sailboat on a blue lake.",
                    "runtime": {"request_phase_graph": graph}}
        request = {"prompt": prompt, "ghost_route": True}
        route = {"capability": "chat", "route_runtime": {"request_phase_graph": graph}}
        response = semantics.truth_gate_response_output_claims(
            response, request_payload=request, route_payload=route,
        )
        assert not response.get("runtime", {}).get("truth_guard")
        image_branch = _image_items(graph["downstream_branches"])[0]
        gap = semantics.build_planner_deferred_follow_up_gap_spec(
            response["output_text"], route_payload=route,
        )
        plan = ollmo_webserver._prepare_late_fill_branch_plan(
            expected_capability="image_generation", artifact_gap={**gap, **image_branch},
            current_payload=response, request_payload=request,
            assistant_message=response["output_text"], source_route_payload=route,
            failed_instance_id=None,
        )
        assert plan["capability"] == "image_generation"
        assert plan["infer_payload"]["prompt"] != WEB_PROMPT
        execution = ollmo_webserver._execute_prepared_late_fill_branch(plan)
        assert harness.calls["image_generation"] == 1
        assert Path(execution["infer_result"]["saved_image_path"]).is_file()
        assert execution["execution_contract"]["branch_id"] == image_branch["branch_id"]
        assert execution["execution_contract"]["phase_id"] == image_branch["phase_id"]
        response = _record_executed_branch(
            ollmo_webserver._LATE_FILL_RUNTIME, response, execution,
        )
        image_artifacts = response["artifacts"]
        assert len([item for item in image_artifacts if item["type"] == "image"]) == 1, image_artifacts
        for branch in graph["downstream_branches"]:
            if branch.get("role") != "text_artifact_output":
                continue
            plan = ollmo_webserver._prepare_late_fill_branch_plan(
                expected_capability="chat", artifact_gap={**gap, **branch},
                current_payload=response, request_payload=request,
                assistant_message=response["output_text"], source_route_payload=route,
                failed_instance_id=None,
            )
            execution = ollmo_webserver._execute_prepared_late_fill_branch(plan)
            assert execution["execution_contract"]["branch_id"] == branch["branch_id"]
            response = _record_executed_branch(
                ollmo_webserver._LATE_FILL_RUNTIME, response, execution,
            )
        assert harness.calls["image_generation"] == 1
        assert harness.calls["chat_materialization"] == 2
        assert len([p for p in harness.artifacts_dir.rglob("*") if p.is_file()]) == 3
        finalized, status = ollmo_webserver._LATE_FILL_RUNTIME.finalize_terminal_materialization_contract(
            response, request_payload=request, route_payload=route, artifact_gap=gap,
            terminal_status="completed",
        )
        assert status == "completed", [
            {key: item.get(key) for key in ("branch_id", "phase_id", "check_kind", "status", "evidence", "reason")}
            for item in finalized.get("late_fill", {}).get("materialization_contract_open_checks", [])
        ]
        assert finalized["late_fill"]["final_materialization_contract_status"] == "fulfilled"
        closure = semantics.build_graph_closure_review(
            finalized.get("output_text", ""), request_payload=request, artifact_payload=finalized,
        )
        assert not [c for c in closure["checks"] if c.get("check_kind") == "linked_artifact_binding"]
        assert closure["status"] == "fulfilled", closure["counts"]

        assert not finalized["late_fill"]["materialization_contract_unmet"]
        artifacts = finalized["artifacts"]
        assert len(artifacts) == len({item["artifact_ref"] for item in artifacts}) == 3
        assert {Path(item["path"]).suffix for item in artifacts} == {".html", ".css", ".png"}
        assert all(item["source_response_id"] == response["id"] for item in artifacts)
        image_path = harness.images_dir / "generated.png"
        png = image_path.read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        chunks = {}
        offset = 8
        while offset < len(png):
            length = struct.unpack_from(">I", png, offset)[0]
            kind = png[offset + 4:offset + 8]
            body = png[offset + 8:offset + 8 + length]
            crc = struct.unpack_from(">I", png, offset + 8 + length)[0]
            assert zlib.crc32(kind + body) == crc
            chunks[kind] = body
            offset += length + 12
        assert offset == len(png)
        assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (2, 1, 8, 2, 0, 0, 0)
        assert zlib.decompress(chunks[b"IDAT"]) == b"\x00\xff\x80\x00\x00\x80\xff"
        assert chunks[b"IEND"] == b""
        producer = finalized["late_fill"]["fill_results"][0]
        assert producer["capability"] == "image_generation"
        assert producer["branch_id"] == image_branch["branch_id"]
        assert producer["phase_id"] == image_branch["phase_id"]
        assert Path(producer["saved_image_path"]).resolve() == image_path.resolve()
        assert producer["artifact_ref"] == next(item["artifact_ref"] for item in artifacts if item["type"] == "image")
        producer_artifacts = producer["artifacts"]
        assert len(producer_artifacts) == 1
        for artifact in producer_artifacts:
            assert artifact["branch_id"] == image_branch["branch_id"]
            assert artifact["phase_id"] == image_branch["phase_id"]
            assert artifact["source_response_id"] == response["id"]
            assert artifact["artifact_ref"] == producer["artifact_ref"]
            assert Path(artifact["path"]).resolve() == image_path.resolve()
        assert [item["capability"] for item in harness.call_records] == ["image_generation", "chat", "chat"]
        for refreshed in finalized["late_fill"]["final_text_artifact_refreshes"]:
            assert hashlib.sha256(Path(refreshed["target_path"]).read_bytes()).hexdigest() == refreshed["file_sha256"]
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.images = []
                self.stylesheets = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "img":
                    self.images.append(attrs["src"])
                elif tag == "link" and attrs.get("rel") == "stylesheet":
                    self.stylesheets.append(attrs["href"])

        links = Links()
        links.feed((harness.web_dir / "index.html").read_text())
        assert len(links.images) == len(links.stylesheets) == 1
        assert (harness.web_dir / links.images[0]).resolve() == image_path.resolve()
        assert (harness.web_dir / links.stylesheets[0]).resolve() == (harness.web_dir / "styles.css").resolve()


def test_ready_image_capability_does_not_turn_a_reference_into_generation():
    prompt = 'HTML references image.png through <img src="image.png">.'
    with _GeneratedWebHarness() as harness:
        assert harness.instances["image_generation"]["runtime_status"]["process_alive"] is True
        payload, status = harness.post_response({
            "response_id": "offline_image_reference_only", "ghost_route": True, "prompt": prompt,
        })
        assert status == 200
        payload, status = harness.get_response("offline_image_reference_only", view="truth")
        assert status == 200
        graph = payload["runtime"]["request_phase_graph"]
        assert not _image_items(graph["output_obligations"])
        assert not _image_items(graph.get("downstream_branches") or [])
        assert harness.calls["image_generation"] == 0
        assert list(harness.images_dir.iterdir()) == []


@pytest.mark.parametrize("placement,missing_link,expects_hero", [
    ("Use a light blue page background and a responsive image.", False, False),
    ("Use a dark background color.", False, False),
    ("Use a light blue page background and a responsive image.", True, False),
    ("Use the generated image as the hero background.", False, True),
    ("Use a generated background image.", False, True),
    ("Use the generated image as the background.", False, True),
])
def test_background_color_does_not_create_hero_closure_work(semantics, tmp_path, placement, missing_link, expects_hero):
    prompt = WEB_PROMPT + " " + placement
    html = tmp_path / "index.html"
    css = tmp_path / "styles.css"
    image = tmp_path / "generated.png"
    html.write_text('<link rel="stylesheet" href="styles.css"><h1>Lake</h1>' +
                    ('' if missing_link else '<img src="generated.png">'))
    css.write_text('body { background-color: lightblue; } img { max-width: 100%; height: auto; }')
    image.write_bytes(b"image existence fixture; no image decoding in this closure test")
    payload = {"output_text": "Prepared files.", "runtime": {"request_phase_graph": _graph(prompt)},
               "artifacts": [{"type": "text", "path": str(html), "extension": "html", "name": "index"},
                             {"type": "text", "path": str(css), "extension": "css", "name": "styles"},
                             {"type": "image", "path": str(image)}]}
    review = semantics.build_graph_closure_review(
        payload["output_text"], request_payload={"prompt": prompt, "ghost_route": True}, artifact_payload=payload,
    )
    checks = [c for c in review["checks"] if c.get("check_kind") == "linked_artifact_binding"]
    assert len(checks) == int(expects_hero or missing_link)
    if checks:
        assert checks[0]["status"] == "pending"
        assert checks[0]["text_artifact_target_path"] == str(css if expects_hero else html)
        assert ("Hero section" in checks[0]["content_payload"]) == expects_hero
