#include "SVDPreprocessGpu.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace {

constexpr float kSpeedOfLight = 299792458.0f;

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        const cudaError_t status__ = (call);                                     \
        if (status__ != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(status__));              \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        const cublasStatus_t status__ = (call);                                  \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error("cuBLAS call failed");                      \
        }                                                                        \
    } while (0)

#define CHECK_CUSOLVER(call)                                                     \
    do {                                                                         \
        const cusolverStatus_t status__ = (call);                                \
        if (status__ != CUSOLVER_STATUS_SUCCESS) {                               \
            throw std::runtime_error("cuSOLVER call failed");                    \
        }                                                                        \
    } while (0)

__global__ void expand_baselines_kernel(
    const float* uvw,
    const float* frequencies_hz,
    float* baselines_col_major,
    std::size_t num_rows,
    std::size_t num_channels
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t num_samples = num_rows * num_channels;
    if (idx >= num_samples) {
        return;
    }

    const std::size_t row = idx % num_rows;
    const std::size_t chan = idx / num_rows;
    const float scale = frequencies_hz[chan] / kSpeedOfLight;

    baselines_col_major[0 * num_samples + idx] = uvw[row * 3 + 0] * scale;
    baselines_col_major[1 * num_samples + idx] = uvw[row * 3 + 1] * scale;
    baselines_col_major[2 * num_samples + idx] = uvw[row * 3 + 2] * scale;
}

__global__ void accumulate_sums_kernel(
    const float* baselines_col_major,
    float* sums,
    std::size_t num_samples
) {
    __shared__ float local_sums[3 * 256];
    const int tid = threadIdx.x;
    const std::size_t idx = blockIdx.x * blockDim.x + tid;

    local_sums[tid + 0 * blockDim.x] = 0.0f;
    local_sums[tid + 1 * blockDim.x] = 0.0f;
    local_sums[tid + 2 * blockDim.x] = 0.0f;

    if (idx < num_samples) {
        local_sums[tid + 0 * blockDim.x] = baselines_col_major[0 * num_samples + idx];
        local_sums[tid + 1 * blockDim.x] = baselines_col_major[1 * num_samples + idx];
        local_sums[tid + 2 * blockDim.x] = baselines_col_major[2 * num_samples + idx];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_sums[tid + 0 * blockDim.x] += local_sums[tid + stride + 0 * blockDim.x];
            local_sums[tid + 1 * blockDim.x] += local_sums[tid + stride + 1 * blockDim.x];
            local_sums[tid + 2 * blockDim.x] += local_sums[tid + stride + 2 * blockDim.x];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&sums[0], local_sums[0 * blockDim.x]);
        atomicAdd(&sums[1], local_sums[1 * blockDim.x]);
        atomicAdd(&sums[2], local_sums[2 * blockDim.x]);
    }
}

__global__ void center_baselines_kernel(
    const float* baselines_col_major,
    const float* means,
    float* centered_col_major,
    std::size_t num_samples
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) {
        return;
    }

    centered_col_major[0 * num_samples + idx] = baselines_col_major[0 * num_samples + idx] - means[0];
    centered_col_major[1 * num_samples + idx] = baselines_col_major[1 * num_samples + idx] - means[1];
    centered_col_major[2 * num_samples + idx] = baselines_col_major[2 * num_samples + idx] - means[2];
}

__global__ void build_basis_kernel(
    const float* eigenvectors_col_major,
    float* basis_row_major,
    float* vin_row_major
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int axis = 0; axis < 3; ++axis) {
        basis_row_major[0 * 3 + axis] = eigenvectors_col_major[axis + 2 * 3];
        basis_row_major[1 * 3 + axis] = eigenvectors_col_major[axis + 1 * 3];
        vin_row_major[0 * 3 + axis] = basis_row_major[0 * 3 + axis];
        vin_row_major[1 * 3 + axis] = basis_row_major[1 * 3 + axis];
    }

    vin_row_major[2 * 3 + 0] =
        vin_row_major[0 * 3 + 1] * vin_row_major[1 * 3 + 2] -
        vin_row_major[0 * 3 + 2] * vin_row_major[1 * 3 + 1];
    vin_row_major[2 * 3 + 1] =
        vin_row_major[0 * 3 + 2] * vin_row_major[1 * 3 + 0] -
        vin_row_major[0 * 3 + 0] * vin_row_major[1 * 3 + 2];
    vin_row_major[2 * 3 + 2] =
        vin_row_major[0 * 3 + 0] * vin_row_major[1 * 3 + 1] -
        vin_row_major[0 * 3 + 1] * vin_row_major[1 * 3 + 0];
}

__global__ void project_bin_kernel(
    const float* baselines_col_major,
    const float* basis_row_major,
    float* bin_row_major,
    std::size_t num_samples
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) {
        return;
    }

    const float x0 = baselines_col_major[0 * num_samples + idx];
    const float x1 = baselines_col_major[1 * num_samples + idx];
    const float x2 = baselines_col_major[2 * num_samples + idx];

    bin_row_major[idx * 2 + 0] =
        x0 * basis_row_major[0 * 3 + 0] +
        x1 * basis_row_major[0 * 3 + 1] +
        x2 * basis_row_major[0 * 3 + 2];
    bin_row_major[idx * 2 + 1] =
        x0 * basis_row_major[1 * 3 + 0] +
        x1 * basis_row_major[1 * 3 + 1] +
        x2 * basis_row_major[1 * 3 + 2];
}

__global__ void align_pca_signs_kernel(
    float* basis_row_major,
    float* vin_row_major,
    float* bin_row_major,
    std::size_t num_samples
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int component = 0; component < 2; ++component) {
            int max_axis = 0;
            float max_value = fabsf(basis_row_major[component * 3 + 0]);
            for (int axis = 1; axis < 3; ++axis) {
                const float candidate = fabsf(basis_row_major[component * 3 + axis]);
                if (candidate > max_value) {
                    max_value = candidate;
                    max_axis = axis;
                }
            }

            if (basis_row_major[component * 3 + max_axis] < 0.0f) {
                for (int axis = 0; axis < 3; ++axis) {
                    basis_row_major[component * 3 + axis] = -basis_row_major[component * 3 + axis];
                    vin_row_major[component * 3 + axis] = -vin_row_major[component * 3 + axis];
                }
                for (std::size_t sample = 0; sample < num_samples; ++sample) {
                    bin_row_major[sample * 2 + component] = -bin_row_major[sample * 2 + component];
                }
            }
        }

        vin_row_major[2 * 3 + 0] =
            vin_row_major[0 * 3 + 1] * vin_row_major[1 * 3 + 2] -
            vin_row_major[0 * 3 + 2] * vin_row_major[1 * 3 + 1];
        vin_row_major[2 * 3 + 1] =
            vin_row_major[0 * 3 + 2] * vin_row_major[1 * 3 + 0] -
            vin_row_major[0 * 3 + 0] * vin_row_major[1 * 3 + 2];
        vin_row_major[2 * 3 + 2] =
            vin_row_major[0 * 3 + 0] * vin_row_major[1 * 3 + 1] -
            vin_row_major[0 * 3 + 1] * vin_row_major[1 * 3 + 0];
    }
}

__global__ void collapse_visibility_kernel(
    const float* vis0_real,
    const float* vis0_imag,
    const float* vis3_real,
    const float* vis3_imag,
    const std::uint8_t* flag0,
    const std::uint8_t* flag3,
    const float* weight0,
    const float* weight3,
    float* vis_real,
    float* vis_imag,
    std::size_t num_samples
) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) {
        return;
    }

    const float w0 = flag0[idx] ? 0.0f : weight0[idx];
    const float w3 = flag3[idx] ? 0.0f : weight3[idx];
    const float w_sum = w0 + w3;

    if (w_sum > 0.0f) {
        vis_real[idx] = (vis0_real[idx] * w0 + vis3_real[idx] * w3) / w_sum;
        vis_imag[idx] = (vis0_imag[idx] * w0 + vis3_imag[idx] * w3) / w_sum;
    } else {
        vis_real[idx] = 0.0f;
        vis_imag[idx] = 0.0f;
    }
}

template <typename T>
T* cuda_alloc_and_copy(const std::vector<T>& host_values) {
    T* device_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&device_ptr, host_values.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(
        device_ptr,
        host_values.data(),
        host_values.size() * sizeof(T),
        cudaMemcpyHostToDevice
    ));
    return device_ptr;
}

}  // namespace

void preprocess_measurement_set_gpu(
    const HostMeasurementSetData& host_data,
    DevicePreprocessBuffers& device_buffers
) {
    if (host_data.num_samples == 0) {
        throw std::runtime_error("Measurement set is empty");
    }

    cublasHandle_t cublas_handle = nullptr;
    cusolverDnHandle_t cusolver_handle = nullptr;

    float* d_uvw = nullptr;
    float* d_freq = nullptr;
    float* d_vis0_real = nullptr;
    float* d_vis0_imag = nullptr;
    float* d_vis3_real = nullptr;
    float* d_vis3_imag = nullptr;
    std::uint8_t* d_flag0 = nullptr;
    std::uint8_t* d_flag3 = nullptr;
    float* d_weight0 = nullptr;
    float* d_weight3 = nullptr;
    float* d_baselines = nullptr;
    float* d_centered = nullptr;
    float* d_sums = nullptr;
    float* d_covariance = nullptr;
    float* d_eigenvalues = nullptr;
    float* d_basis = nullptr;
    int* d_info = nullptr;
    float* d_workspace = nullptr;

    try {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));

        d_uvw = cuda_alloc_and_copy(host_data.uvw);
        d_freq = cuda_alloc_and_copy(host_data.frequencies_hz);
        d_vis0_real = cuda_alloc_and_copy(host_data.vis0_real);
        d_vis0_imag = cuda_alloc_and_copy(host_data.vis0_imag);
        d_vis3_real = cuda_alloc_and_copy(host_data.vis3_real);
        d_vis3_imag = cuda_alloc_and_copy(host_data.vis3_imag);
        d_flag0 = cuda_alloc_and_copy(host_data.flag0);
        d_flag3 = cuda_alloc_and_copy(host_data.flag3);
        d_weight0 = cuda_alloc_and_copy(host_data.weight0);
        d_weight3 = cuda_alloc_and_copy(host_data.weight3);

        const std::size_t num_samples = host_data.num_samples;
        const int threads = 256;
        const int blocks = static_cast<int>((num_samples + threads - 1) / threads);

        CHECK_CUDA(cudaMalloc(&d_baselines, 3 * num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_centered, 3 * num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sums, 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_covariance, 9 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_eigenvalues, 3 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_basis, 6 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_vin, 9 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_bin, 2 * num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_vis_real, num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&device_buffers.d_vis_imag, num_samples * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        expand_baselines_kernel<<<blocks, threads>>>(
            d_uvw,
            d_freq,
            d_baselines,
            host_data.num_rows,
            host_data.num_channels
        );
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemset(d_sums, 0, 3 * sizeof(float)));
        accumulate_sums_kernel<<<blocks, threads>>>(d_baselines, d_sums, num_samples);
        CHECK_CUDA(cudaGetLastError());

        std::vector<float> sums(3);
        CHECK_CUDA(cudaMemcpy(sums.data(), d_sums, 3 * sizeof(float), cudaMemcpyDeviceToHost));
        for (float& value : sums) {
            value /= static_cast<float>(num_samples);
        }
        CHECK_CUDA(cudaMemcpy(d_sums, sums.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

        center_baselines_kernel<<<blocks, threads>>>(d_baselines, d_sums, d_centered, num_samples);
        CHECK_CUDA(cudaGetLastError());

        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            3,
            3,
            static_cast<int>(num_samples),
            &alpha,
            d_centered,
            static_cast<int>(num_samples),
            d_centered,
            static_cast<int>(num_samples),
            &beta,
            d_covariance,
            3
        ));

        int workspace_size = 0;
        CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_UPPER,
            3,
            d_covariance,
            3,
            d_eigenvalues,
            &workspace_size
        ));
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size * sizeof(float)));

        CHECK_CUSOLVER(cusolverDnSsyevd(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_UPPER,
            3,
            d_covariance,
            3,
            d_eigenvalues,
            d_workspace,
            workspace_size,
            d_info
        ));

        int info = 0;
        CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            throw std::runtime_error("cuSOLVER eigen decomposition failed");
        }

        build_basis_kernel<<<1, 1>>>(d_covariance, d_basis, device_buffers.d_vin);
        CHECK_CUDA(cudaGetLastError());

        project_bin_kernel<<<blocks, threads>>>(
            d_baselines,
            d_basis,
            device_buffers.d_bin,
            num_samples
        );
        CHECK_CUDA(cudaGetLastError());

        align_pca_signs_kernel<<<1, 1>>>(
            d_basis,
            device_buffers.d_vin,
            device_buffers.d_bin,
            num_samples
        );
        CHECK_CUDA(cudaGetLastError());

        collapse_visibility_kernel<<<blocks, threads>>>(
            d_vis0_real,
            d_vis0_imag,
            d_vis3_real,
            d_vis3_imag,
            d_flag0,
            d_flag3,
            d_weight0,
            d_weight3,
            device_buffers.d_vis_real,
            device_buffers.d_vis_imag,
            num_samples
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        device_buffers.num_samples = num_samples;
    } catch (...) {
        free_device_preprocess_buffers(device_buffers);
        cudaFree(d_uvw);
        cudaFree(d_freq);
        cudaFree(d_vis0_real);
        cudaFree(d_vis0_imag);
        cudaFree(d_vis3_real);
        cudaFree(d_vis3_imag);
        cudaFree(d_flag0);
        cudaFree(d_flag3);
        cudaFree(d_weight0);
        cudaFree(d_weight3);
        cudaFree(d_baselines);
        cudaFree(d_centered);
        cudaFree(d_sums);
        cudaFree(d_covariance);
        cudaFree(d_eigenvalues);
        cudaFree(d_basis);
        cudaFree(d_info);
        cudaFree(d_workspace);
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
        }
        if (cusolver_handle != nullptr) {
            cusolverDnDestroy(cusolver_handle);
        }
        throw;
    }

    cudaFree(d_uvw);
    cudaFree(d_freq);
    cudaFree(d_vis0_real);
    cudaFree(d_vis0_imag);
    cudaFree(d_vis3_real);
    cudaFree(d_vis3_imag);
    cudaFree(d_flag0);
    cudaFree(d_flag3);
    cudaFree(d_weight0);
    cudaFree(d_weight3);
    cudaFree(d_baselines);
    cudaFree(d_centered);
    cudaFree(d_sums);
    cudaFree(d_covariance);
    cudaFree(d_eigenvalues);
    cudaFree(d_basis);
    cudaFree(d_info);
    cudaFree(d_workspace);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolver_handle));
}

void download_preprocess_outputs(
    const DevicePreprocessBuffers& device_buffers,
    HostPreprocessOutputs& host_outputs
) {
    host_outputs.num_samples = device_buffers.num_samples;
    host_outputs.bin.resize(device_buffers.num_samples * 2);
    host_outputs.vin.resize(9);
    host_outputs.vis_real.resize(device_buffers.num_samples);
    host_outputs.vis_imag.resize(device_buffers.num_samples);

    CHECK_CUDA(cudaMemcpy(
        host_outputs.bin.data(),
        device_buffers.d_bin,
        host_outputs.bin.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaMemcpy(
        host_outputs.vin.data(),
        device_buffers.d_vin,
        host_outputs.vin.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaMemcpy(
        host_outputs.vis_real.data(),
        device_buffers.d_vis_real,
        host_outputs.vis_real.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaMemcpy(
        host_outputs.vis_imag.data(),
        device_buffers.d_vis_imag,
        host_outputs.vis_imag.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

void free_device_preprocess_buffers(DevicePreprocessBuffers& device_buffers) {
    cudaFree(device_buffers.d_bin);
    cudaFree(device_buffers.d_vin);
    cudaFree(device_buffers.d_vis_real);
    cudaFree(device_buffers.d_vis_imag);
    device_buffers.d_bin = nullptr;
    device_buffers.d_vin = nullptr;
    device_buffers.d_vis_real = nullptr;
    device_buffers.d_vis_imag = nullptr;
    device_buffers.num_samples = 0;
}
