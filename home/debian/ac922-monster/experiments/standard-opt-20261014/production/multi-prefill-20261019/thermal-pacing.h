#pragma once

#include <nvml.h>

#include "prefill-thermal-policy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

static prefill_thermal_policy::profile_id production_prefill_thermal_profile() {
    static const bool relaxed = [] {
        const char * value = std::getenv("AC922_PREFILL_THERMAL_RELAXED");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return prefill_thermal_policy::select_profile(relaxed);
}

// The driver is already installed; resolve telemetry without a new link dependency.
struct production_nvml {
    void * library = nullptr;
    decltype(&nvmlDeviceGetTemperature) temperature = nullptr;
    decltype(&nvmlShutdown) shutdown = nullptr;
    std::array<nvmlDevice_t,6> devices{};
    const prefill_thermal_policy::profile_id profile = production_prefill_thermal_profile();
    const prefill_thermal_policy::thresholds limits =
            prefill_thermal_policy::get_thresholds(profile);
    production_nvml() {
        library=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);
        require(library!=nullptr,"thermal pacing NVML library unavailable");
        auto init=reinterpret_cast<decltype(&nvmlInit_v2)>(dlsym(library,"nvmlInit_v2"));
        auto count=reinterpret_cast<decltype(&nvmlDeviceGetCount_v2)>(dlsym(library,"nvmlDeviceGetCount_v2"));
        auto handle=reinterpret_cast<decltype(&nvmlDeviceGetHandleByIndex_v2)>(dlsym(library,"nvmlDeviceGetHandleByIndex_v2"));
        temperature=reinterpret_cast<decltype(temperature)>(dlsym(library,"nvmlDeviceGetTemperature"));
        shutdown=reinterpret_cast<decltype(shutdown)>(dlsym(library,"nvmlShutdown"));
        require(init && count && handle && temperature && shutdown,"thermal pacing NVML interface incomplete");
        require(init()==NVML_SUCCESS,"thermal pacing NVML init failed");
        unsigned n=0;
        require(count(&n)==NVML_SUCCESS && n==devices.size(),"thermal pacing requires six GPUs");
        for (unsigned i=0;i<n;++i) require(handle(i,&devices[i])==NVML_SUCCESS,"thermal pacing GPU handle failed");
        std::lock_guard<std::mutex> lock(runner_log_mutex());
        std::cerr << "prefill-thermal-pacing requested=1 selected=1 sensor=NVML mode="
                  << prefill_thermal_policy::profile_name(profile)
                  << " trigger_c=" << limits.trigger_c
                  << " resume_c=" << limits.resume_c
                  << " hard_stop_c=" << limits.hard_stop_c << '\n';
    }
    ~production_nvml() { if (shutdown) shutdown(); }
};
static double production_gpu_temperature() {
    static production_nvml nvml;
    unsigned peak=0;
    for (auto device:nvml.devices) {
        unsigned value=0;
        require(nvml.temperature(device,NVML_TEMPERATURE_GPU,&value)==NVML_SUCCESS && value>=5 && value<100,
                "thermal pacing GPU sensor read failed");
        peak=std::max(peak,value);
    }
    return double(peak);
}
static int64_t production_prefill_thermal_wait(const std::string & id) {
    if (!production.serving || !env_one("AC922_THERMAL_PACING")) return 0;
    const auto profile = production_prefill_thermal_profile();
    const auto limits = prefill_thermal_policy::get_thresholds(profile);
    double peak=production_gpu_temperature();
    const auto initial_action = prefill_thermal_policy::classify(
            static_cast<int>(peak), profile, prefill_thermal_policy::phase::normal);
    require(initial_action != prefill_thermal_policy::action::hard_stop,
            "GPU temperature guard: reached hard stop");
    if (initial_action == prefill_thermal_policy::action::proceed) return 0;
    const auto start=now_us();
    {
        std::lock_guard<std::mutex> lock(runner_log_mutex());
        std::cerr << "prefill-thermal-pacing wait=1 id=" << id
                  << " mode=" << prefill_thermal_policy::profile_name(profile)
                  << " peak_c=" << peak
                  << " trigger_c=" << limits.trigger_c
                  << " resume_c=" << limits.resume_c
                  << " hard_stop_c=" << limits.hard_stop_c << '\n';
    }
    while (!production.cancelled(id)) {
        const auto wait_action = prefill_thermal_policy::classify(
                static_cast<int>(peak), profile, prefill_thermal_policy::phase::cooling);
        require(wait_action != prefill_thermal_policy::action::hard_stop,
                "GPU temperature guard: reached hard stop");
        if (wait_action == prefill_thermal_policy::action::proceed) break;
        require(now_us()-start<120000000,"thermal pacing cooldown exceeded120seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        peak=production_gpu_temperature();
    }
    return now_us()-start;
}
