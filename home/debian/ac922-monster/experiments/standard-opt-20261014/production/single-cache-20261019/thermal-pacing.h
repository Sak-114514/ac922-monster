#pragma once
#include <nvml.h>

// The driver is already installed; resolve telemetry without a new link dependency.
struct production_nvml {
    void * library = nullptr;
    decltype(&nvmlDeviceGetTemperature) temperature = nullptr;
    decltype(&nvmlShutdown) shutdown = nullptr;
    std::array<nvmlDevice_t,6> devices{};
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
        std::cerr << "prefill-thermal-pacing requested=1 selected=1 sensor=NVML trigger_c=59 resume_c=54 hard_stop_c=65\n";
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
    double peak=production_gpu_temperature();
    require(peak<65,"GPU temperature guard: reached65C");
    if (peak<59) return 0;
    const auto start=now_us();
    {
        std::lock_guard<std::mutex> lock(runner_log_mutex());
        std::cerr << "prefill-thermal-pacing wait=1 id=" << id
                  << " peak_c=" << peak << " resume_c=54 hard_stop_c=65\n";
    }
    while (peak>54 && !production.cancelled(id)) {
        require(now_us()-start<120000000,"thermal pacing cooldown exceeded120seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        peak=production_gpu_temperature();
        require(peak<65,"GPU temperature guard: reached65C");
    }
    return now_us()-start;
}
