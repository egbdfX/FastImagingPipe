#ifndef SVD_PREPROCESS_GPU_H
#define SVD_PREPROCESS_GPU_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct HostMeasurementSetData {
    std::size_t num_rows = 0;
    std::size_t num_channels = 0;
    std::size_t num_samples = 0;
    std::vector<float> uvw;
    std::vector<float> frequencies_hz;
    std::vector<float> vis0_real;
    std::vector<float> vis0_imag;
    std::vector<float> vis3_real;
    std::vector<float> vis3_imag;
    std::vector<std::uint8_t> flag0;
    std::vector<std::uint8_t> flag3;
    std::vector<float> weight0;
    std::vector<float> weight3;
};

struct DevicePreprocessBuffers {
    float* d_bin = nullptr;
    float* d_vin = nullptr;
    float* d_vis_real = nullptr;
    float* d_vis_imag = nullptr;
    std::size_t num_samples = 0;
};

struct HostPreprocessOutputs {
    std::size_t num_samples = 0;
    std::vector<float> bin;
    std::vector<float> vin;
    std::vector<float> vis_real;
    std::vector<float> vis_imag;
};

HostMeasurementSetData read_measurement_set(const std::string& ms_path);
void preprocess_measurement_set_gpu(
    const HostMeasurementSetData& host_data,
    DevicePreprocessBuffers& device_buffers
);
void download_preprocess_outputs(
    const DevicePreprocessBuffers& device_buffers,
    HostPreprocessOutputs& host_outputs
);
void free_device_preprocess_buffers(DevicePreprocessBuffers& device_buffers);

#endif
