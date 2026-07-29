/* Includes */
#include "fip/cpu_kernels.h"



/* Function definitions */
void fip_cpu_kernel_mean_3d       (double         mean[3],
                                   const double (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u=0, v=0, w=0;
    size_t i;

    for(i=0;i<len;i++){
        u += alpha*uvw[i][0];
        v += alpha*uvw[i][1];
        w += alpha*uvw[i][2];
    }

    mean[0] = u/len;
    mean[1] = v/len;
    mean[2] = w/len;
}
void fip_cpu_kernel_mean_3df      (double         mean[3],
                                   const float  (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u=0, v=0, w=0;
    size_t i;

    for(i=0;i<len;i++){
        u += alpha*uvw[i][0];
        v += alpha*uvw[i][1];
        w += alpha*uvw[i][2];
    }

    mean[0] = u/len;
    mean[1] = v/len;
    mean[2] = w/len;
}

void fip_cpu_kernel_covariance_3d (double         cov [3][3],
                                   const double (*uvw)[3],
                                   const double   mean[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u, v, w;
    register double M00=0,
                    M10=0, M11=0,
                    M20=0, M21=0, M22=0;
    size_t i;

    for(i=0;i<len;i++){
        u = alpha*uvw[i][0] - mean[0];
        v = alpha*uvw[i][1] - mean[1];
        w = alpha*uvw[i][2] - mean[2];

        M00 += u*u;
        M10 += v*u;
        M11 += v*v;
        M20 += w*u;
        M21 += w*v;
        M22 += w*w;
    }

    cov[0][0] =             M00;
    cov[1][0] = cov[0][1] = M10;
    cov[1][1] =             M11;
    cov[2][0] = cov[0][2] = M20;
    cov[2][1] = cov[1][2] = M21;
    cov[2][2] =             M22;
}
void fip_cpu_kernel_covariance_3df(double         cov [3][3],
                                   const float  (*uvw)[3],
                                   const double   mean[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u, v, w;
    register double M00=0,
                    M10=0, M11=0,
                    M20=0, M21=0, M22=0;
    size_t i;

    for(i=0;i<len;i++){
        u = alpha*uvw[i][0] - mean[0];
        v = alpha*uvw[i][1] - mean[1];
        w = alpha*uvw[i][2] - mean[2];

        M00 += u*u;
        M10 += v*u;
        M11 += v*v;
        M20 += w*u;
        M21 += w*v;
        M22 += w*w;
    }

    cov[0][0] =             M00;
    cov[1][0] = cov[0][1] = M10;
    cov[1][1] =             M11;
    cov[2][0] = cov[0][2] = M20;
    cov[2][1] = cov[1][2] = M21;
    cov[2][2] =             M22;
}

void fip_cpu_kernel_svd_3x3       (double         R[3][3],
                                   const double   M[3][3],
                                   const double   tolerance,
                                   const size_t   max_iters){
    (void)R;
    (void)M;
    (void)tolerance;
    (void)max_iters;
}

void fip_cpu_kernel_transform_2d  (float        (*out)[2],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const double (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u, v, w;
    register double R00=R[0][0], R01=R[0][1], R02=R[0][2],
                    R10=R[1][0], R11=R[1][1], R12=R[1][2];
    size_t i;

    for(i=0;i<len;i++){
        u = alpha*uvw[i][0] - mean[0];
        v = alpha*uvw[i][1] - mean[1];
        w = alpha*uvw[i][2] - mean[2];

        out[i][0] = R00*u + R01*v + R02*w;
        out[i][1] = R10*u + R11*v + R12*w;
    }
}
void fip_cpu_kernel_transform_3d  (float        (*out)[3],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const double (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u, v, w;
    register double R00=R[0][0], R01=R[0][1], R02=R[0][2],
                    R10=R[1][0], R11=R[1][1], R12=R[1][2],
                    R20=R[2][0], R21=R[2][1], R22=R[2][2];
    size_t i;

    for(i=0;i<len;i++){
        u = alpha*uvw[i][0] - mean[0];
        v = alpha*uvw[i][1] - mean[1];
        w = alpha*uvw[i][2] - mean[2];

        out[i][0] = R00*u + R01*v + R02*w;
        out[i][1] = R10*u + R11*v + R12*w;
        out[i][2] = R20*u + R21*v + R22*w;
    }
}
void fip_cpu_kernel_transform_2df (float        (*out)[2],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const float  (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u, v, w;
    register double R00=R[0][0], R01=R[0][1], R02=R[0][2],
                    R10=R[1][0], R11=R[1][1], R12=R[1][2];
    size_t i;

    for(i=0;i<len;i++){
        u = alpha*uvw[i][0] - mean[0];
        v = alpha*uvw[i][1] - mean[1];
        w = alpha*uvw[i][2] - mean[2];

        out[i][0] = R00*u + R01*v + R02*w;
        out[i][1] = R10*u + R11*v + R12*w;
    }
}
void fip_cpu_kernel_transform_3df (float        (*out)[3],
                                   const double   R[3][3],
                                   const double   mean[3],
                                   const float  (*uvw)[3],
                                   const double   alpha,
                                   const size_t   len){
    register double u, v, w;
    register double R00=R[0][0], R01=R[0][1], R02=R[0][2],
                    R10=R[1][0], R11=R[1][1], R12=R[1][2],
                    R20=R[2][0], R21=R[2][1], R22=R[2][2];
    size_t i;

    for(i=0;i<len;i++){
        u = alpha*uvw[i][0] - mean[0];
        v = alpha*uvw[i][1] - mean[1];
        w = alpha*uvw[i][2] - mean[2];

        out[i][0] = R00*u + R01*v + R02*w;
        out[i][1] = R10*u + R11*v + R12*w;
        out[i][2] = R20*u + R21*v + R22*w;
    }
}
