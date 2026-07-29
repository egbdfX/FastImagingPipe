/* Include Guard */
#ifndef INCLUDE_FIP_CPU_KERNELS_H
#define INCLUDE_FIP_CPU_KERNELS_H



/* Includes */
#include <stdlib.h>
#include "fip/visibility.h"



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


LIBFIP_PUBLIC
void fip_cpu_kernel_mean_3d       (double         mean[3],
                                   const double (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len);
LIBFIP_PUBLIC
void fip_cpu_kernel_mean_3df      (double         mean[3],
                                   const float  (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len);

LIBFIP_PUBLIC
void fip_cpu_kernel_covariance_3d (double         cov [3][3],
                                   const double (*uvw)[3],
                                   const double   mean[3],
                                   const double   alpha,
                                   const size_t   len);
LIBFIP_PUBLIC
void fip_cpu_kernel_covariance_3df(double         cov [3][3],
                                   const float  (*uvw)[3],
                                   const double   mean[3],
                                   const double   alpha,
                                   const size_t   len);

LIBFIP_PUBLIC
void fip_cpu_kernel_svd_3x3       (double         R[3][3],
                                   const double   M[3][3],
                                   const double   tolerance,
                                   const size_t   max_iters);

LIBFIP_PUBLIC
void fip_cpu_kernel_transform_2d  (float        (*out)[2],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const double (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len);
LIBFIP_PUBLIC
void fip_cpu_kernel_transform_3d  (float        (*out)[3],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const double (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len);
LIBFIP_PUBLIC
void fip_cpu_kernel_transform_2df (float        (*out)[2],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const float  (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len);
LIBFIP_PUBLIC
void fip_cpu_kernel_transform_3df (float        (*out)[3],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const float  (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
