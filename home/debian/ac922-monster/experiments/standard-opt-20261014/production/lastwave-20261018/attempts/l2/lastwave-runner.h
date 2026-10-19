#pragma once

// Included after bank: one coordinator mutates only drained bank owners.
static json lastwave_flow_record(const bank & b, const lane & l) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(b.target.get()));
    return {{"id", l.id}, {"bank", b.id}, {"input_tokens", l.input_tokens},
        {"output_tokens", l.generated.size()}, {"cached_tokens", l.cached_tokens},
        {"output", common_detokenize(vocab, l.generated, false)},
        {"token_ids", l.generated}, {"emitted_us", l.emitted},
        {"proposed", l.proposed}, {"accepted", l.accepted},
        {"prefill_us", l.prefill_us}, {"target_prefill_us", l.target_prefill_us},
        {"mtp_prefill_us", l.mtp_prefill_us}, {"thermal_wait_us", l.thermal_wait_us},
        {"prefill_started_us", b.prefill_started_us}, {"prefill_finished_us", b.prefill_finished_us},
        {"warmup_started_us", b.warmup_started_us}, {"warmup_finished_us", b.warmup_finished_us},
        {"decode_started_us", b.generation_started_us}, {"decode_finished_us", b.generation_finished_us},
        {"warmup_frames", b.request_warmup_frames}, {"active_chain_lanes", b.active_chain_lanes_total},
        {"fallback_lanes", b.fallback_lanes_total}};
}

static json run_lastwave_requests(const std::vector<bank *> & banks,
        const std::string & wave_id, bool refill_enabled,
        const std::function<json(const json &)> & prepare_request,
        const std::string & logit_dir, bool dump_before_mtp, bool full_logits,
        int warmup_frames, int request_limit) {
    const int64_t started = now_us();
    std::array<bool, 4> idle_offered{};
    json flows = json::array();
    size_t admitted_refills = 0;
    bool closing = false;
    int64_t closing_started = 0;
    const size_t max_refills = 32;
    auto submit = [&](bank * b) {
        b->submit_request_owner(wave_id, logit_dir, dump_before_mtp, full_logits, warmup_frames);
    };
    for (bank * b : banks) if (!b->lanes.empty()) submit(b);
    try {
        for (;;) {
            bool busy = false;
            for (size_t i = 0; i < banks.size(); ++i) {
                bank * b = banks[i];
                if (b->owner_loop_submitted && b->completion.valid() &&
                        b->completion.wait_for(std::chrono::microseconds(0)) == std::future_status::ready) {
                    b->join_owner_loop();
                    for (auto & l : b->lanes) {
                        b->api_finish(l);
                        flows.push_back(lastwave_flow_record(*b, l));
                        production.finish_request(l.id);
                    }
                    std::cerr << "lastwave-bank-drained wave=" << wave_id << " bank=" << b->id
                              << " prefill_begin_us=" << b->prefill_started_us
                              << " prefill_end_us=" << b->prefill_finished_us
                              << " decode_begin_us=" << b->generation_started_us
                              << " decode_end_us=" << b->generation_finished_us << '\n';
                    b->lanes.clear();
                }
                busy = busy || b->owner_loop_submitted;
                if (refill_enabled && !closing && admitted_refills < max_refills &&
                        !b->owner_loop_submitted && !idle_offered[i] && b->params.n_ubatch == 256) {
                    production.emit({{"type", "bank_idle"}, {"wave_id", wave_id}, {"bank", b->id},
                        {"capacity", b->params.n_ctx}, {"ubatch", 256}});
                    idle_offered[i] = true;
                }
            }
            json command;
            while (refill_enabled && production.poll_refill(wave_id, command)) {
                const std::string request_id = command.at("requests").at(0).value("id", "");
                const int index = command.at("bank").get<int>();
                bool compatible = index >= 0 && size_t(index) < banks.size();
                bank * destination = compatible ? banks[size_t(index)] : nullptr;
                compatible = compatible && !destination->owner_loop_submitted &&
                        destination->lanes.empty() && destination->params.n_ubatch == 256 &&
                        admitted_refills < max_refills;
                json request;
                if (compatible) {
                    try {
                        request = prepare_request(command.at("requests").at(0));
                        const size_t reserve = request.at("reserved_tokens").get<size_t>();
                        compatible = reserve <= size_t(request_limit) &&
                            request.at("allocation_tokens").get<size_t>() + 4 <=
                                    size_t(destination->params.n_ctx);
                    } catch (const std::exception & error) {
                        production.emit({{"type", "error"}, {"id", request_id},
                            {"status", 400}, {"message", error.what()}});
                        production.finish_request(request_id);
                        continue;
                    }
                }
                if (!compatible) {
                    production.emit({{"type", "refill_rejected"}, {"wave_id", wave_id},
                        {"bank", index}, {"id", request_id}, {"reason", "capacity_or_geometry"}});
                    production.finish_request(request_id);
                    continue;
                }
                destination->reset(/*keep_prefix=*/true);
                lane l;
                l.id = request_id;
                l.limit = request.at("max_tokens").get<int>();
                l.ignore_eos = request.value("ignore_eos", false);
                l.allow_prefix_cache = request.value("cache_prompt", true);
                l.prompt = request.at("tokens").get<llama_tokens>();
                l.input_tokens = l.prompt.size();
                l.prompt_source = "production_refill";
                destination->lanes.push_back(std::move(l));
                production.emit({{"type", "accepted"}, {"id", request_id},
                    {"input_tokens", request.at("tokens").size()},
                    {"reserved_tokens", request.at("reserved_tokens")},
                    {"bank", index}, {"wave_id", wave_id}, {"refill", true}});
                ++admitted_refills;
                idle_offered[size_t(index)] = false;
                submit(destination);
                busy = true;
            }
            if (!busy) {
                if (!refill_enabled) break;
                if (!closing) {
                    production.request_wave_close(wave_id);
                    closing = true;
                    closing_started = now_us();
                }
                if (production.close_requested(wave_id)) break;
                require(now_us()-closing_started < 10000000,
                        "API did not acknowledge drained wave close within 10 seconds");
            }
            if (production.cancelled("")) {
                // EOF/shutdown marks every active request cancelled. Owners
                // observe it at the next natural frame/chunk boundary.
                if (!busy) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (...) {
        const auto failure = std::current_exception();
        for (auto * b : banks) b->drain_after_failure();
        std::rethrow_exception(failure);
    }
    const int64_t finished = now_us();
    int64_t first_emit = finished, last_emit = started;
    int64_t first_prefill = finished, last_prefill = started;
    uint64_t output_tokens = 0, input_tokens = 0, cached_tokens = 0;
    int64_t target_us = 0, mtp_us = 0, thermal_us = 0;
    for (const auto & flow : flows) {
        output_tokens += flow.at("output_tokens").get<uint64_t>();
        input_tokens += flow.at("input_tokens").get<uint64_t>();
        cached_tokens += flow.at("cached_tokens").get<uint64_t>();
        target_us += flow.at("target_prefill_us").get<int64_t>();
        mtp_us += flow.at("mtp_prefill_us").get<int64_t>();
        thermal_us += flow.at("thermal_wait_us").get<int64_t>();
        first_prefill = std::min(first_prefill, flow.at("prefill_started_us").get<int64_t>());
        last_prefill = std::max(last_prefill, flow.at("prefill_finished_us").get<int64_t>());
        for (const auto & emitted : flow.at("emitted_us")) {
            first_emit = std::min(first_emit, emitted.get<int64_t>());
            last_emit = std::max(last_emit, emitted.get<int64_t>());
        }
    }
    json geometry = json::array();
    for (const bank * b : banks) geometry.push_back({{"bank", b->id},
        {"target_context", llama_n_ctx(b->target.get())},
        {"draft_context", llama_n_ctx(b->draft_init->context())},
        {"target_ubatch", b->params.n_ubatch}});
    return {{"wave_id", wave_id}, {"mode", "bank_local_phase_progression"}, {"banks", geometry},
        {"refill_count", admitted_refills}, {"flow_count", flows.size()}, {"flows", flows},
        {"wall_seconds", (finished-started)/1e6},
        {"whole_wave_output_tok_s", output_tokens*1e6/std::max<int64_t>(1, finished-started)},
        {"all_emissions_tok_s", output_tokens*1e6/std::max<int64_t>(1,last_emit-first_emit)},
        {"prefill_input_tokens", input_tokens}, {"prefill_wall_seconds", (last_prefill-first_prefill)/1e6},
        {"cached_tokens", cached_tokens}, {"prefill_evaluated_tokens", input_tokens-cached_tokens},
        {"uncached_prefill", cached_tokens == 0},
        {"prefill_aggregate_wall_tok_s", (input_tokens-cached_tokens)*1e6/std::max<int64_t>(1,last_prefill-first_prefill)},
        {"logical_prompt_tok_s", input_tokens*1e6/std::max<int64_t>(1,last_prefill-first_prefill)},
        {"prefill_scope", admitted_refills ? "mixed decode and refilled prefill; not raw prefill" : "initial independent bank prefills"},
        {"target_prefill_us_sum", target_us}, {"mtp_prefill_us_sum", mtp_us},
        {"thermal_wait_us_sum", thermal_us}, {"mtp_depth", 3}};
}
