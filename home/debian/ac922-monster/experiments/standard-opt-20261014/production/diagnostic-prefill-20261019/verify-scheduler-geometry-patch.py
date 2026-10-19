#!/usr/bin/env python3
"""Static scope/lifecycle check for scheduler-geometry.patch."""

from pathlib import Path
import re


patch_path = Path(__file__).with_name("scheduler-geometry.patch")
text = patch_path.read_text()
paths = set(re.findall(r"^--- a/(.+)$", text, flags=re.MULTILINE))
expected = {"src/llama-context.cpp", "ggml/src/ggml-backend.cpp"}
assert paths == expected, f"unexpected patch paths: {sorted(paths)}"
assert ".cu" not in text and ".h |" not in text

context_section = text.split("--- a/src/llama-context.cpp\n", 1)[1].split(
        "--- a/ggml/src/ggml-backend.cpp\n", 1)[0]
backend_section = text.split("--- a/ggml/src/ggml-backend.cpp\n", 1)[1]

# The context profile is registered at construction, used around every
# sched_reserve scheduler creation, and removed through constructor/destructor
# lifetime guards without changing llama_context's class layout.
for required in (
        "llama_sched_layer_geometry_profiles",
        "llama_sched_layer_geometry_registration scheduler_geometry_registration(this)",
        "scheduler_geometry_registration.commit()",
        "llama_sched_layer_geometry_unregister scheduler_geometry_unregister(this)",
        "llama_sched_layer_geometry_scope geometry_scope(this)",
):
    assert required in context_section, f"missing context lifecycle hook: {required}"

# Geometry override is thread-local and scoped; callers outside a scope still
# read the environment, while a captured-unset value masks later env changes.
for required in (
        "thread_local std::vector<ggml_backend_sched_layer_geometry_profile>",
        "ggml_backend_sched_layer_geometry_push",
        "ggml_backend_sched_layer_geometry_pop",
        "return std::getenv(name);",
        "profile.copies_present ? profile.copies.c_str() : nullptr",
        "profile.host_slots_present ? profile.host_slots.c_str() : nullptr",
):
    assert required in backend_section, f"missing thread-profile behavior: {required}"

assert backend_section.count("ggml_backend_sched_layer_geometry_env(\n") >= 3, \
        "copy-depth and both pinned-slot reads must use the scoped profile"
assert 'extern "C" GGML_API int ggml_backend_sched_layer_geometry_' not in backend_section, \
        "GGML_API expands to extern; do not place it after extern-C linkage"
assert 'GGML_API int ggml_backend_sched_layer_geometry_push(' in backend_section, \
        "exported prototypes must carry GGML_API inside the existing extern-C block"
assert 'extern "C" int ggml_backend_sched_layer_geometry_push(' in backend_section, \
        "definitions must use prior exported prototypes without repeating GGML_API"
print("scheduler geometry patch static check: PASS (2 host C++ files, scoped context lifetime, env fallback)")
