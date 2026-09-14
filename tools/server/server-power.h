#pragma once

#include <memory>

struct common_params;

// holds max performance power settings while the server has work, releases them after --power-switch-idle seconds without work
// GPU: NVIDIA PowerMizer mode 1 through NVML, CPU: power-profiles-daemon performance hold (Linux only)
struct server_power {
    virtual ~server_power() = default;

    // called from the server loop, do not block
    virtual void set_busy() = 0;
    virtual void set_idle() = 0;
};

// returns nullptr unless --power-switch-gpu or --power-switch-cpu is set
std::unique_ptr<server_power> server_power_init(const common_params & params);
