from pathlib import Path
import hashlib,shutil
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OLD=BASE/'production/candidate-20261018'
source=OLD/'bank-worker.cpp'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='33aad78786a6ebd8e410e27f16d96e596e16c6d7f9fd0a4ad4cdb70223f803e7'
s=source.read_text()
def replace(old,new):
    global s
    assert s.count(old)==1,(s.count(old),old[:90]);s=s.replace(old,new)
replace('        const float layer_split_ratio[6] = {10.0f, 10.0f, 9.0f, 7.0f, 8.0f, 6.0f};', '        const float layer_split_ratio[6] = {9.0f, 9.0f, 8.0f, 8.0f, 8.0f, 8.0f};')
replace('        auto cp = common_context_params_to_llama(params);','''        const bool wide_prefill = params.n_ubatch >= 512;
        params.n_batch = wide_prefill ? 32768 : 4096;
        require(params.n_ubatch == 256 || params.n_ubatch == 512 || params.n_ubatch == 1024,
                "unsupported combined target UBatch");
        require(setenv("GGML_SCHED_LAYER_DEVICE_COPIES", wide_prefill ? "8" : "0", 1) == 0 &&
                setenv("GGML_SCHED_LAYER_PINNED_HOST_SLOTS", wide_prefill ? "8" : "4", 1) == 0,
                "cannot configure drained scheduler copy geometry");
        auto cp = common_context_params_to_llama(params);''')
replace('            draft_init = common_speculative_init_from_params(params, model, target.get());','''            auto draft_params = params;
            draft_params.n_ubatch = 256;
            draft_init = common_speculative_init_from_params(draft_params, model, target.get());''')
replace('            const size_t chunk_limit = std::min<size_t>((size_t) params.n_batch, 16384);','            const size_t chunk_limit = std::min<size_t>((size_t) params.n_batch, 32768);')
replace('            require(chunk_limit == 4096 &&','            require(chunk_limit == (params.n_ubatch >= 512 ? 32768 : 4096) &&')
replace('        require(prefill_kv_pad == 8192 || prefill_kv_pad == 65536,','        require(prefill_kv_pad == 8192 || prefill_kv_pad == 32768 || prefill_kv_pad == 65536,')
replace('        require(prefill_batch == 4096,','        require(prefill_batch == 32768,')
replace('        require(env_int("LLAMA_MTP_N_UBATCH", 0) == 512,','        require(env_int("LLAMA_MTP_N_UBATCH", 0) == 256,')
replace('                    bank_params.n_ctx = desired[id];','''                    bank_params.n_ctx = desired[id];
                        bank_params.n_ubatch = selected_ubatch[id];
                        bank_params.cpuparams.n_threads = selected_ubatch[id] >= 512 ? 16 : 2;
                        bank_params.cpuparams_batch.n_threads = selected_ubatch[id] >= 512 ? 16 : 8;''')
replace('                const auto desired = production_shape::choose(needed, production_pool);\n                if (production_shape::rebuild(existing, desired, production_pool)) {','''                const bool single_prefill = scenario.at("requests").size() == 1 &&
                        scenario.at("requests").at(0).at("tokens").size() >= 8192;
                std::array<uint32_t,4> selected_ubatch{256,256,256,256};
                if (single_prefill) {
                    const auto & single_request=scenario.at("requests").at(0);
                    const size_t input_size=std::max(single_request.at("tokens").size(),
                            single_request.at("allocation_tokens").get<size_t>() - single_request.at("max_tokens").get<size_t>());
                    selected_ubatch[0] = input_size <= 49152 ? 1024 : 512;
                    needed[0] = std::max<uint32_t>(needed[0],66048);
                }
                auto desired = production_shape::choose(needed, production_pool);
                if (single_prefill) for (size_t i=1;i<desired.size();++i) desired[i]=1024;
                bool profile_changed = false;
                for (size_t i=0;i<banks.size();++i) {
                    profile_changed = profile_changed || banks[i]->params.n_ubatch != selected_ubatch[i];
                }
                std::cerr << "production-combo-mode single_prefill=" << single_prefill
                          << " target_ubatch=" << selected_ubatch[0] << " draft_ubatch=256"
                          << " logical_batch=" << (single_prefill ? 32768 : 4096) << " copies=" << (single_prefill ? 8 : 4)
                          << " runtime=R444 helper=R5 graph_lease_ms=60000 quality_route=standard\\n";
                if (profile_changed || production_shape::rebuild(existing, desired, production_pool)) {''')
replace('                for (const auto & request : scenario.at("requests")) {\n                    production.emit', '                require(setenv("GGML_SCHED_LAYER_DEVICE_COPIES", single_prefill ? "8" : "0", 1) == 0 &&\n                        setenv("GGML_SCHED_LAYER_PINNED_HOST_SLOTS", single_prefill ? "8" : "4", 1) == 0,\n                        "cannot keep active scheduler profile for late reserve");\n                for (const auto & request : scenario.at("requests")) {\n                    production.emit')
replace('#include "production-shape-policy.h"', '#include "production-shape-policy.h"\n#include "thermal-pacing.h"')
replace('    int64_t prefill_us = 0;', '    int64_t prefill_us = 0;\n    int64_t thermal_wait_us = 0;')
replace('                if (production.cancelled(l.id)) { l.done = true; break; }', '                if (l.prompt.size() >= 8192) l.thermal_wait_us += production_prefill_thermal_wait(l.id);\n                if (production.cancelled(l.id)) { l.done = true; break; }')
# Preserve a metrics-only record in the worker protocol for bounded acceptance runs.
replace('                production.finish_wave(api_wave.at("id"));\n            } else {','''                json timing;
                for (const char * key : {"prefill_input_tokens", "prefill_wall_seconds",
                        "prefill_aggregate_wall_tok_s", "target_prefill_us_sum", "mtp_prefill_us_sum",
                        "single_request_target_prefill_tok_s", "shared_tg", "all_emissions_tok_s"}) {
                    if (record.contains(key)) timing[key]=record.at(key);
                }
                production.finish_wave(api_wave.at("id"),timing);
                std::cerr << "production-combo-metrics wave=" << api_wave.at("id").get<std::string>()
                          << " values=" << timing.dump() << '\\n';
            } else {''')
replace('{"allocation_tokens", allocation_tokens}, {"ignore_eos", false}', '{"allocation_tokens", allocation_tokens}, {"ignore_eos", request.value("ignore_eos", false)}')
# Internal corpus requests are used by the frozen benchmark checker, not HTTP clients.
replace('                        if (request.value("raw", false)) {','''                        if (request.contains("corpus_token_count")) {
                            const size_t begin=request.value("corpus_token_start",size_t(0));
                            const size_t count=request.at("corpus_token_count").get<size_t>();
                            require(count && begin <= v45_corpus_tokens.size() && count <= v45_corpus_tokens.size()-begin,
                                    "internal corpus slice out of range");
                            tokens.assign(v45_corpus_tokens.begin()+begin,v45_corpus_tokens.begin()+begin+count);
                        } else if (request.value("raw", false)) {''')
# Reuse the established V412 long semantic pad construction.
replace('                            tokens = common_tokenize(v45_vocab, rendered.prompt, true, true);','''                            tokens = common_tokenize(v45_vocab, rendered.prompt, true, true);
                            if (request.contains("semantic_pad_to_tokens")) {
                                require(input.messages.size()==1, "semantic pad requires one controlled message");
                                const auto & content_text=input.messages.front().content;
                                const size_t content_begin=rendered.prompt.find(content_text);
                                require(content_begin!=std::string::npos &&
                                        rendered.prompt.find(content_text,content_begin+1)==std::string::npos,
                                        "semantic content must appear once");
                                auto before=common_tokenize(v45_vocab,rendered.prompt.substr(0,content_begin),true,true);
                                const auto content=common_tokenize(v45_vocab,content_text,false,true);
                                const auto after=common_tokenize(v45_vocab,rendered.prompt.substr(content_begin+content_text.size()),false,true);
                                const auto separator=common_tokenize(v45_vocab,"\\n\\n",false,true);
                                const size_t fixed=before.size()+content.size()+after.size()+separator.size();
                                const size_t total=request.at("semantic_pad_to_tokens").get<size_t>();
                                require(total>=fixed && total-fixed<=v45_corpus_tokens.size(), "semantic padding out of range");
                                before.insert(before.end(),v45_corpus_tokens.begin(),v45_corpus_tokens.begin()+total-fixed);
                                before.insert(before.end(),separator.begin(),separator.end());
                                before.insert(before.end(),content.begin(),content.end());
                                before.insert(before.end(),after.begin(),after.end());
                                tokens=std::move(before);
                            }''')
replace('                json timing;','                int64_t thermal_wait=0;\n                for (const auto * owner : banks) for (const auto & l : owner->lanes) thermal_wait+=l.thermal_wait_us;\n                json timing={{"thermal_wait_us_sum",thermal_wait}};')
(HERE/'bank-worker-combo.cpp').write_text(s)
for h in OLD.glob('*.h'):shutil.copy2(h,HERE/h.name)
h=(HERE/'production-ipc.h').read_text().replace('void finish_wave(const std::string & id) {','void finish_wave(const std::string & id, const json & timing=json::object()) {').replace('emit({{"type", "wave_done"}, {"id", id}});','emit({{"type", "wave_done"}, {"id", id}, {"timings", timing}});')
(HERE/'production-ipc.h').write_text(h)
print('COMBO_SOURCE_READY',hashlib.sha256(s.encode()).hexdigest())
