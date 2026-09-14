#include "server-power.h"
#include "server-common.h"

#include "common.h"
#include "log.h"

#include <memory>

// returns true if a side is enabled; warns about power flags that have no effect
static bool power_check_flags(const common_params & params) {
    if (!params.power_switch_gpu && !params.power_switch_cpu) {
        const common_params defaults{};
        if (params.power_switch_idle != defaults.power_switch_idle || params.power_switch_check > 0) {
            SRV_WRN("%s", "power switch: --power-switch-idle/--power-switch-check given without --power-switch-gpu or --power-switch-cpu; feature off\n");
        }
        return false;
    }
    if (params.power_switch_check > 0 && !params.power_switch_cpu) {
        SRV_WRN("%s", "power switch: --power-switch-check needs --power-switch-cpu; ignored\n");
    }
    return true;
}

#if defined(__linux__)

#include "ggml.h"
#include "subproc.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

namespace {

constexpr unsigned int POWER_NVML_MODE_ADAPTIVE = 0;
constexpr unsigned int POWER_NVML_MODE_MAX_PERF = 1;
constexpr int64_t      POWER_HOLD_START_CHECK_MS = 2000;

constexpr int POWER_SUBPROC_OPTIONS = subprocess_option_no_window | subprocess_option_search_user_path |
                                      subprocess_option_inherit_environment | subprocess_option_combined_stdout_stderr;

// mirror of nvmlPowerMizerModes_v1_t
struct power_nvml_modes {
    unsigned int current_mode;
    unsigned int mode;
    unsigned int supported;
};

// NVML is loaded at runtime, so the build has no NVML dependency
// the mode is a request of this process: other NVML clients keep their own, and the driver drops ours when the process exits
struct power_nvml {
    void * lib = nullptr;

    int          (*init)(void)                               = nullptr;
    int          (*shutdown)(void)                           = nullptr;
    const char * (*error_string)(int)                        = nullptr;
    int          (*device_count)(unsigned int *)             = nullptr;
    int          (*device_handle)(unsigned int, void **)     = nullptr;
    int          (*device_name)(void *, char *, unsigned int) = nullptr;
    int          (*get_mode)(void *, power_nvml_modes *)     = nullptr;
    int          (*set_mode)(void *, power_nvml_modes *)     = nullptr;

    std::vector<void *> devices;

    bool load(std::string & names) {
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            SRV_WRN("power switch: cannot load libnvidia-ml.so.1 (%s), GPU side disabled\n", dlerror());
            return false;
        }

        init          = (int (*)(void))                               dlsym(lib, "nvmlInit_v2");
        shutdown      = (int (*)(void))                               dlsym(lib, "nvmlShutdown");
        error_string  = (const char * (*)(int))                       dlsym(lib, "nvmlErrorString");
        device_count  = (int (*)(unsigned int *))                     dlsym(lib, "nvmlDeviceGetCount_v2");
        device_handle = (int (*)(unsigned int, void **))              dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        device_name   = (int (*)(void *, char *, unsigned int))       dlsym(lib, "nvmlDeviceGetName");
        get_mode      = (int (*)(void *, power_nvml_modes *))         dlsym(lib, "nvmlDeviceGetPowerMizerMode_v1");
        set_mode      = (int (*)(void *, power_nvml_modes *))         dlsym(lib, "nvmlDeviceSetPowerMizerMode_v1");

        if (!init || !shutdown || !error_string || !device_count || !device_handle || !device_name || !get_mode || !set_mode) {
            SRV_WRN("%s", "power switch: libnvidia-ml.so.1 has no PowerMizer functions, GPU side disabled\n");
            dlclose(lib);
            lib = nullptr;
            return false;
        }

        int rc = init();
        if (rc != 0) {
            SRV_WRN("power switch: nvmlInit failed (%s), GPU side disabled\n", error_string(rc));
            dlclose(lib);
            lib = nullptr;
            return false;
        }

        unsigned int count = 0;
        rc = device_count(&count);
        if (rc != 0) {
            SRV_WRN("power switch: cannot count GPUs (%s)\n", error_string(rc));
            count = 0;
        }

        for (unsigned int i = 0; i < count; i++) {
            void * dev = nullptr;
            char name[96] = {0};
            power_nvml_modes modes = {0, 0, 0};

            rc = device_handle(i, &dev);
            if (rc == 0) {
                rc = device_name(dev, name, sizeof(name));
            }
            if (rc == 0) {
                rc = get_mode(dev, &modes);
            }
            if (rc != 0) {
                SRV_WRN("power switch: cannot query GPU %u (%s), skipped\n", i, error_string(rc));
                continue;
            }
            if (!((modes.supported >> POWER_NVML_MODE_MAX_PERF) & 1)) {
                SRV_WRN("power switch: GPU %u (%s) does not support PowerMizer mode 1 (supported 0x%x), skipped\n", i, name, modes.supported);
                continue;
            }

            devices.push_back(dev);
            names += string_format("%s%s (0x%x)", names.empty() ? "" : ", ", name, modes.supported);
        }

        if (devices.empty()) {
            SRV_WRN("%s", "power switch: no GPU can use PowerMizer mode 1, GPU side disabled\n");
            shutdown();
            return false;
        }

        return true;
    }

    // returns the number of GPUs set
    int set_all(unsigned int mode) {
        int n = 0;
        for (size_t i = 0; i < devices.size(); i++) {
            power_nvml_modes modes = {0, mode, 0};
            const int rc = set_mode(devices[i], &modes);
            if (rc != 0) {
                SRV_WRN("power switch: set PowerMizer mode %u on GPU %zu failed (%s)\n", mode, i, error_string(rc));
                continue;
            }
            n++;
        }
        return n;
    }

    void unload() {
        // shutdown drops our mode request; no dlclose after init, the library can keep internal state
        if (!devices.empty()) {
            shutdown();
            devices.clear();
        }
    }
};

std::string power_read_all(FILE * f) {
    std::string out;
    if (!f) {
        return out;
    }
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    return out;
}

std::string power_last_line(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    const size_t pos = s.rfind('\n');
    return pos == std::string::npos ? s : s.substr(pos + 1);
}

struct server_power_linux : server_power {
    bool    use_gpu;
    bool    use_cpu;
    int64_t idle_ms;
    int64_t check_ms;

    power_nvml  nvml;
    std::string hold_reason;

    // shared with the server loop
    std::mutex              mtx;
    std::condition_variable cv;
    bool    want_busy = false;
    bool    stop      = false;
    int64_t t_idle    = 0;

    std::thread worker;

    // worker thread only
    bool    applied      = false;
    int     n_gpu_set    = 0;
    int64_t t_hold_check = INT64_MAX;
    bool    hold_started = false;
    bool    hold_warned  = false;
    std::unique_ptr<common_subproc> hold;

    explicit server_power_linux(const common_params & params)
        : use_gpu (params.power_switch_gpu),
          use_cpu (params.power_switch_cpu),
          idle_ms (params.power_switch_idle * 1000LL),
          check_ms(params.power_switch_cpu ? params.power_switch_check * 1000LL : 0) {
        std::string names;
        if (use_gpu && !nvml.load(names)) {
            use_gpu = false;
        }
        if (use_cpu && !common_subproc::is_supported()) {
            SRV_WRN("%s", "power switch: this build has no subprocess support, CPU side disabled\n");
            use_cpu = false;
        }
        hold_reason = string_format("llama-server pid %d", (int) getpid());

        if (use_gpu || use_cpu) {
            const std::string gpu   = use_gpu ? "on: " + names : "off";
            const std::string check = check_ms > 0 ? string_format("every %d s", params.power_switch_check) : "off";
            SRV_INF("power switch: gpu %s, cpu %s, idle %d s, hold check %s\n", gpu.c_str(), use_cpu ? "on" : "off", params.power_switch_idle, check.c_str());
        }
    }

    ~server_power_linux() override {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
        if (applied) {
            const int64_t t_ms = release();
            SRV_INF("power switch: exit -> released in %" PRId64 " ms\n", t_ms);
        }
        nvml.unload();
    }

    void start() {
        worker = std::thread([this]() { run(); });
    }

    void set_busy() override {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (want_busy) {
                return;
            }
            want_busy = true;
        }
        cv.notify_one();
    }

    void set_idle() override {
        {
            std::lock_guard<std::mutex> lock(mtx);
            // repeated idle scans (e.g. /metrics polls) must not move the deadline
            if (!want_busy) {
                return;
            }
            want_busy = false;
            t_idle    = ggml_time_ms();
        }
        cv.notify_one();
    }

    // compares the wanted state with the applied state, so no stale operation can run
    void run() {
        std::unique_lock<std::mutex> lock(mtx);
        while (!stop) {
            const int64_t now       = ggml_time_ms();
            const bool    busy      = want_busy;
            const int64_t t_release = t_idle + idle_ms;

            if (busy && !applied) {
                lock.unlock();
                const int64_t t_ms = apply();
                SRV_INF("power switch: busy -> applied gpu=%d cpu=%s in %" PRId64 " ms\n", n_gpu_set, hold ? "on" : "off", t_ms);
                lock.lock();
                continue;
            }

            if (!busy && applied && now >= t_release) {
                const int64_t idle_s = (now - t_idle) / 1000;
                lock.unlock();
                const int64_t t_ms = release();
                SRV_INF("power switch: idle %" PRId64 " s -> released in %" PRId64 " ms\n", idle_s, t_ms);
                lock.lock();
                continue;
            }

            if (hold && now >= t_hold_check) {
                lock.unlock();
                check_hold();
                lock.lock();
                continue;
            }

            int64_t deadline = INT64_MAX;
            if (!busy && applied) {
                deadline = t_release;
            }
            if (hold) {
                deadline = std::min(deadline, t_hold_check);
            }
            if (deadline == INT64_MAX) {
                cv.wait(lock);
            } else {
                cv.wait_for(lock, std::chrono::milliseconds(deadline - now));
            }
        }
    }

    // returns the time spent in ms
    int64_t apply() {
        const int64_t t0 = ggml_time_ms();
        n_gpu_set = use_gpu ? nvml.set_all(POWER_NVML_MODE_MAX_PERF) : 0;
        if (use_cpu) {
            hold_start();
        }
        applied = true;
        return ggml_time_ms() - t0;
    }

    // returns the time spent in ms
    int64_t release() {
        const int64_t t0 = ggml_time_ms();
        if (hold) {
            std::string last;
            const int code = hold_finish(last);
            if (last.find("No hold with cookie") != std::string::npos) {
                SRV_WRN("%s", "power switch: the CPU hold was dropped by another program (e.g. a manual profile change)\n");
            } else if (code != 0) {
                SRV_WRN("power switch: powerprofilesctl exited with code %d: %s\n", code, last.c_str());
            }
        }
        if (n_gpu_set > 0) {
            nvml.set_all(POWER_NVML_MODE_ADAPTIVE);
            n_gpu_set = 0;
        }
        applied = false;
        return ggml_time_ms() - t0;
    }

    void hold_start() {
        // "cat" reads stdin until llama-server closes it or exits; then powerprofilesctl releases the hold, also after a crash
        const std::vector<std::string> args = {
            "powerprofilesctl", "launch", "-p", "performance", "-i", "llama-server", "-r", hold_reason, "--", "cat",
        };
        hold = std::make_unique<common_subproc>();
        if (!hold->create(args, POWER_SUBPROC_OPTIONS)) {
            SRV_WRN("%s", "power switch: cannot start powerprofilesctl, CPU side disabled\n");
            hold.reset();
            use_cpu = false;
            return;
        }
        t_hold_check = ggml_time_ms() + POWER_HOLD_START_CHECK_MS;
        hold_started = false;
        hold_warned  = false;
    }

    // ends the hold and returns the exit code of powerprofilesctl
    int hold_finish(std::string & last) {
        hold->close_stdin();
        last = power_last_line(power_read_all(hold->stdout_file()));
        const int code = hold->join();
        hold.reset();
        t_hold_check = INT64_MAX;
        return code;
    }

    void check_hold() {
        if (!hold->alive()) {
            std::string last;
            const int code = hold_finish(last);
            SRV_WRN("power switch: powerprofilesctl exited early with code %d, CPU side not held: %s\n", code, last.c_str());
            return;
        }
        if (hold_started && !hold_warned && !hold_listed()) {
            SRV_WRN("%s", "power switch: the CPU hold is gone, dropped by another program (e.g. a manual profile change)\n");
            hold_warned = true;
        }
        hold_started = true;
        t_hold_check = check_ms > 0 ? ggml_time_ms() + check_ms : INT64_MAX;
    }

    // true if power-profiles-daemon lists our hold; also true if busctl cannot run (checks stop)
    bool hold_listed() {
        const std::vector<std::string> args = {
            "busctl", "get-property", "org.freedesktop.UPower.PowerProfiles", "/org/freedesktop/UPower/PowerProfiles",
            "org.freedesktop.UPower.PowerProfiles", "ActiveProfileHolds",
        };
        common_subproc proc;
        if (!proc.create(args, POWER_SUBPROC_OPTIONS)) {
            SRV_WRN("%s", "power switch: cannot run busctl, CPU hold checks disabled\n");
            check_ms = 0;
            return true;
        }
        proc.close_stdin();
        const std::string out  = power_read_all(proc.stdout_file());
        const int         code = proc.join();
        if (code != 0) {
            SRV_WRN("power switch: busctl exited with code %d, CPU hold checks disabled: %s\n", code, power_last_line(out).c_str());
            check_ms = 0;
            return true;
        }
        return out.find("\"" + hold_reason + "\"") != std::string::npos;
    }
};

} // namespace

std::unique_ptr<server_power> server_power_init(const common_params & params) {
    if (!power_check_flags(params)) {
        return nullptr;
    }
    auto power = std::make_unique<server_power_linux>(params);
    if (!power->use_gpu && !power->use_cpu) {
        return nullptr;
    }
    power->start();
    return power;
}

#else

std::unique_ptr<server_power> server_power_init(const common_params & params) {
    if (power_check_flags(params)) {
        SRV_WRN("%s", "power switch: only supported on Linux, ignored\n");
    }
    return nullptr;
}

#endif
