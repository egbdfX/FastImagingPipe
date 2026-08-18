#include <cuda_runtime.h>
#include <cufft.h>
#include <fitsio.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_CUFFT(call)                                                       \
    do {                                                                        \
        cufftResult err__ = (call);                                             \
        if (err__ != CUFFT_SUCCESS) {                                           \
            std::fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, \
                         static_cast<int>(err__));                              \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct CubeLayout {
    long dims[3];
    int time_axis;
    int y_axis;
    int x_axis;
    long n_time;
    long ny;
    long nx;
};

static void fits_die(int status, const char *what) {
    if (status) {
        std::fprintf(stderr, "%s failed:\n", what);
        fits_report_error(stderr, status);
        std::exit(1);
    }
}

static CubeLayout make_layout(const long dims[3], int requested_time_axis) {
    CubeLayout layout{};
    std::copy(dims, dims + 3, layout.dims);

    if (requested_time_axis >= 0) {
        layout.time_axis = requested_time_axis;
    } else {
        layout.time_axis = 0;
        for (int a = 0; a < 3; ++a) {
            if (dims[a] != dims[(a + 1) % 3] && dims[a] != dims[(a + 2) % 3]) {
                layout.time_axis = a;
                break;
            }
        }
    }

    int other[2], n_other = 0;
    for (int a = 0; a < 3; ++a) {
        if (a != layout.time_axis) other[n_other++] = a;
    }
    layout.y_axis = other[0];
    layout.x_axis = other[1];
    layout.n_time = dims[layout.time_axis];
    layout.ny = dims[layout.y_axis];
    layout.nx = dims[layout.x_axis];
    return layout;
}

static void read_fits_cube(const char *path, std::vector<float> *cube, long dims[3]) {
    fitsfile *fptr = nullptr;
    int status = 0, bitpix = 0, naxis = 0;
    long naxes[3] = {1, 1, 1};

    fits_open_file(&fptr, path, READONLY, &status);
    fits_die(status, "fits_open_file");
    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    fits_die(status, "fits_get_img_param");
    if (naxis != 3) {
        std::fprintf(stderr, "Expected a 3D FITS image, got NAXIS=%d\n", naxis);
        std::exit(1);
    }

    dims[0] = naxes[0];
    dims[1] = naxes[1];
    dims[2] = naxes[2];
    long nelem = dims[0] * dims[1] * dims[2];
    cube->resize(static_cast<size_t>(nelem));

    int anynul = 0;
    fits_read_img(fptr, TFLOAT, 1, nelem, nullptr, cube->data(), &anynul, &status);
    fits_die(status, "fits_read_img");
    fits_close_file(fptr, &status);
    fits_die(status, "fits_close_file");
}

static void write_result_fits(const char *path, const std::vector<float> &result,
                              long nx, long ny, long max_x, long max_y,
                              long min_snapshot, float max_eta) {
    fitsfile *fptr = nullptr;
    int status = 0;
    std::string out = "!" + std::string(path);
    long naxes[2] = {nx, ny};

    fits_create_file(&fptr, out.c_str(), &status);
    fits_die(status, "fits_create_file");
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    fits_die(status, "fits_create_img");
    fits_write_img(fptr, TFLOAT, 1, nx * ny, const_cast<float *>(result.data()), &status);
    fits_die(status, "fits_write_img");

    fits_update_key(fptr, TLONG, "MAXX", &max_x, "1-based x index of max result cell", &status);
    fits_update_key(fptr, TLONG, "MAXY", &max_y, "1-based y index of max result cell", &status);
    fits_update_key(fptr, TLONG, "MINSNAP", &min_snapshot,
                    "1-based snapshot index of minimum IQA in max cell", &status);
    fits_update_key(fptr, TFLOAT, "MAXETA", &max_eta, "maximum harmonic-summed score", &status);
    fits_die(status, "fits_update_key");

    fits_close_file(fptr, &status);
    fits_die(status, "fits_close_file");
}

__global__ void prepare_sequences(const float *cube, cufftComplex *seq, int *min_idx,
                                  long d0, long d1, long d2, int time_axis,
                                  int y_axis, int x_axis, long n_time, long ny,
                                  long nx) {
    long cell = blockIdx.x * blockDim.x + threadIdx.x;
    long cells = ny * nx;
    if (cell >= cells) return;

    long y = cell / nx;
    long x = cell - y * nx;
    double sum = 0.0;
    float min_val = INFINITY;
    int local_min = 0;

    for (long t = 0; t < n_time; ++t) {
        long coord[3];
        coord[time_axis] = t;
        coord[y_axis] = y;
        coord[x_axis] = x;
        long idx = coord[0] + d0 * (coord[1] + d1 * coord[2]);
        float v = cube[idx];
        sum += static_cast<double>(v);
        if (v < min_val) {
            min_val = v;
            local_min = static_cast<int>(t);
        }
    }

    float mean = static_cast<float>(sum / static_cast<double>(n_time));
    for (long t = 0; t < n_time; ++t) {
        long coord[3];
        coord[time_axis] = t;
        coord[y_axis] = y;
        coord[x_axis] = x;
        long idx = coord[0] + d0 * (coord[1] + d1 * coord[2]);
        float w = 1.0f;
        if (n_time > 1) {
            w = 0.54f - 0.46f * cosf(2.0f * 3.14159265358979323846f *
                                     static_cast<float>(t) /
                                     static_cast<float>(n_time - 1));
        }
        seq[cell * n_time + t].x = (cube[idx] - mean) * w;
        seq[cell * n_time + t].y = 0.0f;
    }
    min_idx[cell] = local_min;
}

__device__ float p1_value(const cufftComplex *fft, long cell, long n_time, long k) {
    cufftComplex z = fft[cell * n_time + k];
    float scale = 1.0f / static_cast<float>(n_time);
    float p = (z.x * scale) * (z.x * scale) + (z.y * scale) * (z.y * scale);
    long half = n_time / 2;
    long L = half + 1;
    if (k > 0 && k < L - 1) p *= 2.0f;
    return p;
}

__global__ void analyze_fft(const cufftComplex *fft, float *result, long n_time,
                            long cells, int max_harmonic) {
    long cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cells) return;

    long L = n_time / 2 + 1;
    float best = -INFINITY;

    for (long k = 0; k < L; ++k) {
        float x_new = (L > 1) ? static_cast<float>(k) / static_cast<float>(L - 1) : 0.0f;
        float p_sum = 0.0f;

        for (int h = 1; h <= max_harmonic; ++h) {
            long comp_len = L / h;
            if (comp_len <= 0) continue;
            float pos = (comp_len > 1) ? x_new * static_cast<float>(comp_len - 1) : 0.0f;
            long lo = static_cast<long>(floorf(pos));
            long hi = min(lo + 1, comp_len - 1);
            float frac = pos - static_cast<float>(lo);
            float vlo = p1_value(fft, cell, n_time, lo * h);
            float vhi = p1_value(fft, cell, n_time, hi * h);
            p_sum += vlo + frac * (vhi - vlo);
        }

        if (p_sum > best) best = p_sum;
    }

    result[cell] = best;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s input_iqa.fits output_result.fits [--time-axis 1|2|3] "
                 "[--max-harmonic N]\n",
                 argv0);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    int requested_time_axis = -1;
    int max_harmonic = 19;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--time-axis") == 0 && i + 1 < argc) {
            requested_time_axis = std::atoi(argv[++i]) - 1;
            if (requested_time_axis < 0 || requested_time_axis > 2) {
                std::fprintf(stderr, "--time-axis must be 1, 2, or 3\n");
                return 1;
            }
        } else if (std::strcmp(argv[i], "--max-harmonic") == 0 && i + 1 < argc) {
            max_harmonic = std::atoi(argv[++i]);
            if (max_harmonic < 1) {
                std::fprintf(stderr, "--max-harmonic must be positive\n");
                return 1;
            }
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    std::vector<float> cube;
    long dims[3];
    read_fits_cube(argv[1], &cube, dims);
    CubeLayout layout = make_layout(dims, requested_time_axis);
    long cells = layout.ny * layout.nx;

    std::printf("Input FITS axes: [%ld, %ld, %ld]\n", dims[0], dims[1], dims[2]);
    std::printf("Using time axis %d: n_time=%ld, result shape=%ld x %ld\n",
                layout.time_axis + 1, layout.n_time, layout.ny, layout.nx);

    float *d_cube = nullptr, *d_result = nullptr;
    cufftComplex *d_seq = nullptr;
    int *d_min_idx = nullptr;

    CHECK_CUDA(cudaMalloc(&d_cube, cube.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_seq, static_cast<size_t>(cells * layout.n_time) * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_result, static_cast<size_t>(cells) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_min_idx, static_cast<size_t>(cells) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_cube, cube.data(), cube.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto output_path_start = std::chrono::steady_clock::now();

    int block = 128;
    int grid = static_cast<int>((cells + block - 1) / block);
    prepare_sequences<<<grid, block>>>(d_cube, d_seq, d_min_idx, dims[0], dims[1], dims[2],
                                       layout.time_axis, layout.y_axis, layout.x_axis,
                                       layout.n_time, layout.ny, layout.nx);
    CHECK_CUDA(cudaGetLastError());

    cufftHandle plan;
    int n[1] = {static_cast<int>(layout.n_time)};
    CHECK_CUFFT(cufftPlanMany(&plan, 1, n, nullptr, 1, static_cast<int>(layout.n_time),
                              nullptr, 1, static_cast<int>(layout.n_time),
                              CUFFT_C2C, static_cast<int>(cells)));
    CHECK_CUFFT(cufftExecC2C(plan, d_seq, d_seq, CUFFT_FORWARD));

    analyze_fft<<<grid, block>>>(d_seq, d_result, layout.n_time, cells, max_harmonic);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> result(static_cast<size_t>(cells));
    std::vector<int> min_idx(static_cast<size_t>(cells));
    CHECK_CUDA(cudaMemcpy(result.data(), d_result, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(min_idx.data(), d_min_idx, min_idx.size() * sizeof(int), cudaMemcpyDeviceToHost));

    auto max_it = std::max_element(result.begin(), result.end());
    long max_cell = static_cast<long>(std::distance(result.begin(), max_it));
    long max_y = max_cell / layout.nx;
    long max_x = max_cell - max_y * layout.nx;
    long min_snapshot_1based = static_cast<long>(min_idx[max_cell]) + 1;
    float max_eta = *max_it;

    write_result_fits(argv[2], result, layout.nx, layout.ny, max_x + 1, max_y + 1,
                      min_snapshot_1based, max_eta);
    auto output_path_stop = std::chrono::steady_clock::now();
    double output_path_ms =
        std::chrono::duration<double, std::milli>(output_path_stop - output_path_start).count();

    std::printf("Wrote %s\n", argv[2]);
    std::printf("Max result cell: x=%ld, y=%ld (1-based)\n", max_x + 1, max_y + 1);
    std::printf("Max eta: %.9g\n", max_eta);
    std::printf("Minimum-IQA snapshot at max cell: %ld (1-based)\n", min_snapshot_1based);
    std::printf("CUDA output-path time: %.6f ms\n", output_path_ms);

    cufftDestroy(plan);
    cudaFree(d_cube);
    cudaFree(d_seq);
    cudaFree(d_result);
    cudaFree(d_min_idx);
    return 0;
}
