/* Include Guard */
#ifndef INCLUDE_FIP_CUDA_KERNELS_H
#define INCLUDE_FIP_CUDA_KERNELS_H


/* Includes */
#include <stdlib.h>
#include <cuda_runtime.h>
#include "fip/visibility.h"


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


LIBFIP_PUBLIC
void fip_cuda_kernel_convkernel    (dim3                cuda_grid,
                                    dim3                cuda_thrd,
                                    size_t              cuda_shmem,
                                    cudaStream_t        cuda_stream,
                                    float*              conv_corr_kernel,
                                    size_t              image_size,
                                    size_t              grid_size,
                                    float               conv_corr_norm_factor);

LIBFIP_PUBLIC
void fip_cuda_kernel_gridding      (dim3                cuda_grid,
                                    dim3                cuda_thrd,
                                    size_t              cuda_shmem,
                                    cudaStream_t        cuda_stream,
                                    void*               grid,
                                    const void*         visibilities,
                                    const float*        coords,
                                    const float         transform[3][3],
                                    const size_t        grid_stride,
                                    const size_t        grid_size,
                                    const size_t        num_baselines,
                                    const float         r1r2_scale);

LIBFIP_PUBLIC
void fip_cuda_kernel_interp          (dim3                cuda_grid,
                                    dim3                cuda_thrd,
                                    size_t              cuda_shmem,
                                    cudaStream_t        cuda_stream,
                                    float*              image,
                                    const void*         grid,
                                    const float         transform[3][3],
                                    const size_t        image_stride,
                                    const size_t        image_size,
                                    const size_t        grid_stride,
                                    const size_t        grid_size,
                                    const float         dc_rad,
                                    const float*        conv_corr_kernel,
                                    const float         conv_corr_norm_factor,
                                    const float         inv_num_baselines);

LIBFIP_PUBLIC
void fip_cuda_kernel_tlisi         (dim3                cuda_grid,
                                    dim3                cuda_thrd,
                                    size_t              cuda_shmem,
                                    cudaStream_t        cuda_stream,
                                    float*              result,
                                    const float*        image0,
                                    const float*        max0,
                                    const float*        image1,
                                    const float*        max1,
                                    const float*        image2,
                                    const float*        max2,
                                    const size_t        result_stride,
                                    const size_t        image_stride,
                                    const size_t        image_size,
                                    const size_t        unit_size,
                                    const size_t        unit_num,
                                    const float         C,
                                    const int           big_endian);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
