from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
source = BASE / 'fixtures/fourbank-frame-log-v453-root-20261018/dual-context-v221-vocab-fourbank-d3.cpp'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'ebe5568c4c7d554b32fdcf556445b739d852908870c9ad9d10595cf4a6356f06'
s = source.read_text()
def replace(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:100])
    s = s.replace(old, new)

replace('struct lane {', '#include "production-ipc.h"\n#include "production-shape-policy.h"\n\nstruct lane {')
replace('            require(chunk_limit == 4096 &&\n                    (l.prompt.size() <= chunk_limit || l.prompt.size() % chunk_limit == 0),', '            require(chunk_limit == 4096 &&\n                    (production.serving || l.prompt.size() <= chunk_limit || l.prompt.size() % chunk_limit == 0),')
replace('                const size_t chunk_end = std::min(chunk_begin + chunk_limit, l.prompt.size());', '                if (production.cancelled(l.id)) { l.done = true; break; }\n                const size_t chunk_end = std::min(chunk_begin + chunk_limit, l.prompt.size());')
replace('            require(row >= 0, "V50 prompt produced no final target row");', '            if (l.done && production.serving) continue;\n            require(row >= 0, "V50 prompt produced no final target row");')
replace('                require(b->target_kv_shape.context_per_sequence == 17152 &&', '                require(b->target_kv_shape.context_per_sequence == (live_wave ? b->params.n_ctx : 17152) &&')
replace('                        b->draft_kv_shape.context_per_sequence == 17152 &&', '                        b->draft_kv_shape.context_per_sequence == (live_wave ? b->params.n_ctx : 17152) &&')
replace('    bool ignore_eos = false;', '    bool ignore_eos = false;\n    bool completion_sent = false;')
replace('    bool active() const {\n', '''    void api_finish(lane & l) {
        if (!production.serving || !l.done || l.completion_sent) return;
        l.completion_sent = true;
        const std::string reason = production.cancelled(l.id) ? "cancelled" :
                int(l.generated.size()) >= l.limit ? "length" : "stop";
        production.emit({{"type", "done"}, {"id", l.id}, {"input_tokens", l.input_tokens},
            {"output_tokens", l.generated.size()}, {"finish_reason", reason}});
    }
    bool active() {
        if (production.serving) for (auto & l : lanes) {
            if (production.cancelled(l.id)) l.done = true;
            api_finish(l);
        }
''')
replace('            l.generated.push_back(l.current);\n', '            l.generated.push_back(l.current);\n            production.token(l.id, vocab, l.current);\n')
replace('                l.generated.push_back(tok);\n', '                l.generated.push_back(tok);\n                production.token(l.id, vocab, tok);\n')
replace('            if (spec) common_speculative_begin(spec.get(), seq, l.prompt);', '            if (spec) common_speculative_begin(spec.get(), seq, l.prompt);\n            api_finish(l);')
replace('            l.done = l.done || int(l.generated.size()) >= l.limit;', '            l.done = l.done || int(l.generated.size()) >= l.limit;\n            api_finish(l);')
replace('    try {\n        const char * kv_unified_text', '''    try {
        production.configure();
        const int production_pool = env_int("AC922_POOL_TOKENS", 262144);
        const int production_limit = env_int("AC922_REQUEST_TOKENS", 200000);
        require(!production.enabled() || (production_limit >= 1024 && production_limit <= production_pool &&
                production_pool <= 1048576), "invalid production context limits");
        const int production_floor_mib = env_int("AC922_MIN_GPU_FREE_MIB", 256);
        require(production_floor_mib >= 256 && production_floor_mib <= 4096, "invalid GPU free memory floor");
        const char * kv_unified_text''')
replace('        require(bool(model), "model load failed");', '        require(bool(model), "model load failed");\n        require(!production.enabled() || production_limit + 4 <= (int) llama_model_n_ctx_train(model.get()), "production request limit exceeds native model context");')
replace('        for (const auto & scenario : scenarios) {\n', '''        for (size_t production_index = 0; ; ++production_index) {
            json scenario, api_wave;
            const bool live_wave = production_index >= scenarios.size();
            if (!live_wave) scenario = scenarios.at(production_index);
            else {
                if (!production.enabled() || !production.next(api_wave)) break;
                scenario = {{"label", api_wave.at("id")}, {"kind", "production"}, {"dual", true},
                            {"requests", json::array()}};
                try {
                    const auto & requests = api_wave.at("requests");
                    require(requests.is_array() && !requests.empty() && requests.size() <= 4,
                            "a wave must contain 1 to 4 requests");
                    size_t reserved = 0;
                    for (const auto & request : requests) {
                        try {
                        const int output = request.at("max_tokens").get<int>();
                        require(output > 0 && output <= 8192, "max_tokens must be 1..8192");
                        llama_tokens tokens;
                        if (request.value("raw", false)) {
                            tokens = common_tokenize(v45_vocab, request.at("prompt").get<std::string>(), true, false);
                        } else {
                            common_chat_templates_inputs input;
                            input.enable_thinking = false;
                            if (request.contains("messages")) {
                                for (const auto & item : request.at("messages")) {
                                    common_chat_msg message;
                                    message.role = item.at("role").get<std::string>();
                                    require(message.role == "system" || message.role == "user" || message.role == "assistant",
                                            "unsupported message role");
                                    message.content = item.at("content").get<std::string>();
                                    input.messages.push_back(std::move(message));
                                }
                            } else {
                                common_chat_msg message;
                                message.role = "user";
                                message.content = request.at("prompt").get<std::string>();
                                input.messages.push_back(std::move(message));
                            }
                            require(!input.messages.empty(), "messages cannot be empty");
                            const auto rendered = common_chat_templates_apply(templates.get(), input);
                            tokens = common_tokenize(v45_vocab, rendered.prompt, true, true);
                        }
                        const size_t tokens_reserved = tokens.size() + (size_t) output;
                        const size_t allocation_tokens = request.value("capacity_tokens", tokens_reserved);
                        require(allocation_tokens >= tokens_reserved && allocation_tokens <= (size_t) production_limit,
                                "invalid context reservation");
                        require(!tokens.empty() && tokens_reserved <= (size_t) production_limit,
                                "request exceeds maximum prompt plus output context");
                        reserved += ((allocation_tokens + 4 + 255) / 256) * 256;
                        scenario["requests"].push_back({{"id", request.at("id")}, {"tokens", tokens},
                            {"max_tokens", output}, {"reserved_tokens", tokens_reserved}, {"allocation_tokens", allocation_tokens}, {"ignore_eos", false}});
                        } catch (const std::exception & error) {
                            production.emit({{"type", "error"}, {"id", request.value("id", "")},
                                             {"status", 400}, {"message", error.what()}});
                        }
                    }
                    if (scenario["requests"].empty()) {
                        production.finish_wave(api_wave.at("id"));
                        continue;
                    }
                    require(reserved + (4 - scenario["requests"].size()) * 1024 <= (size_t) production_pool, "shared context pool exceeded");
                } catch (const std::exception & error) {
                    for (const auto & request : scenario.at("requests")) {
                        production.emit({{"type", "error"}, {"id", request.value("id", "")},
                                         {"status", 400}, {"message", error.what()}});
                    }
                    production.finish_wave(api_wave.at("id"));
                    continue;
                }
            }
''')
# Run the frozen native checker, then skip its benchmark descriptor in service mode.
replace('            std::cerr << "scenario-start label=" << scenario_label\n', '''            if (production.enabled() && !live_wave && !semantic_scenario) {
                require(native_semantics_checked, "production ready before native gate");
                production.start(production_limit, production_pool);
                continue;
            }
            if (live_wave) {
                // Repartition only after all owners and target/draft work have drained.
                for (auto * b : banks) b->reset();
                production_shape::capacities needed{}, existing{};
                for (size_t i = 0; i < banks.size(); ++i) {
                    existing[i] = banks[i]->params.n_ctx;
                    if (i < scenario.at("requests").size()) {
                        const auto & request = scenario.at("requests").at(i);
                        needed[i] = (uint32_t) ((request.at("allocation_tokens").get<size_t>() + 4 + 255) / 256 * 256);
                    }
                }
                const auto desired = production_shape::choose(needed, production_pool);
                if (production_shape::rebuild(existing, desired, production_pool)) {
                    banks.clear();
                    bank_storage.clear();
                    for (int id = 0; id < 4; ++id) {
                        auto shape = kv_layout;
                        shape.bank_context = shape.context_per_sequence = desired[id];
                        common_params bank_params = p;
                        bank_params.n_ctx = desired[id];
                        bank_storage.emplace_back(std::make_unique<bank>(id, bank_params, model.get(),
                                serialize_cpu_submit, shape));
                        banks.push_back(bank_storage.back().get());
                    }
                    for (int i = 0; i < llama_model_n_devices(model.get()); ++i) {
                        size_t free_bytes = 0, total_bytes = 0;
                        ggml_backend_dev_memory(llama_model_get_device(model.get(), i), &free_bytes, &total_bytes);
                        require(free_bytes >= (size_t) production_floor_mib * 1024 * 1024,
                                "production context allocation fell below configured GPU memory floor");
                    }
                    std::cerr << "production-context-repartition pool=" << production_pool << " bank_contexts=";
                    for (const auto * b : banks) std::cerr << b->params.n_ctx << ',';
                    std::cerr << " shared_model=1 drained_boundary=1\\n";
                }
                for (const auto & request : scenario.at("requests")) {
                    production.emit({{"type", "accepted"}, {"id", request.at("id")},
                        {"input_tokens", request.at("tokens").size()},
                        {"reserved_tokens", request.at("reserved_tokens")}});
                }
            }
            std::cerr << "scenario-start label=" << scenario_label
''')
replace('            results.push_back(record);\n            std::ofstream(results_path) << results.dump(2) << \'\\n\';\n', '''            if (live_wave) {
                for (auto * b : banks) for (auto & l : b->lanes) b->api_finish(l);
                production.finish_wave(api_wave.at("id"));
            } else {
                results.push_back(record);
                std::ofstream(results_path) << results.dump(2) << '\\n';
            }
''')
(HERE/'bank-worker.cpp').write_text(s)
print('WORKER_PREPARED', hashlib.sha256(s.encode()).hexdigest())
