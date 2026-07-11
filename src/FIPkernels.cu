#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <npp.h>

#include "fits-utils.h"
#include "fip-pipeline-cuda-state.h"


/**
 * A C/C++ compiler will emit a diagnostic on a redefinion of a macro to something
 * that isn't effectively the same thing after already having been defined [1].
 * This includes redefining M_PI to a different number of decimals.
 *
 * Unfortunately, certain low-quality operating systems don't define M_PI.
 * We borrow the CUDA Toolkit's definition of it if it's available.
 *
 * [1] https://gcc.gnu.org/onlinedocs/cpp/Undefining-and-Redefining-Macros.html
 */

#if   !defined(M_PI)
# if   defined CUDART_PI
#  define M_PI CUDART_PI
# else
#  define M_PI 3.14159265358979323846
# endif
#endif


/* The gridding kernels are developed based on SKA SDP (https://gitlab.com/ska-telescope/sdp/ska-sdp-func). */

__constant__ float quadrature_nodes[14] = {
	0.9964425,0.98130317,0.95425928,0.91563303,0.86589252,
	0.80564137,0.73561088,0.65665109,0.56972047,0.47587422,
	0.37625152,0.27206163,0.16456928,0.05507929
};
__constant__ float quadrature_weights[14] = {
	0.00912428,0.02113211,0.03290143,0.04427293,0.05510735,
	0.06527292,0.07464621,0.08311342,0.09057174,0.09693066,
	0.10211297,0.10605577,0.10871119,0.11004701
};
__constant__ float quadrature_kernel[14] = {
	7.71381676e-07,4.06901586e-06,2.09164257e-05,1.01923695e-04,
	4.61199576e-04,1.90183990e-03,7.02391280e-03,2.28652529e-02,
	6.46725327e-02,1.56933676e-01,3.23208771e-01,5.60024174e-01,
	8.10934691e-01,9.76937533e-01
};


/**
 * @brief Ceiling Divide.
 *
 * Perform a/b, rounding up.
 *
 * @param [in]  a  Dividend.
 * @param [in]  b  Divisor. Undefined behaviour if 0.
 * @return Quotient, rounded up to nearest integer.
 */

__host__ __device__ size_t ceiling_divide(size_t a, size_t b) {
    size_t q =  a/b;
    return q + (a > q*b);
}


__device__ float exp_semicircle(const float beta, const float x){
    const float xx = x*x;
    return xx > 1.0f ? 0.0f : expf(beta*(sqrtf(1.0f - xx) - 1.0f));
}

__global__ void fip_pipe_cuda_convkernel(float* conv_corr_kernel,
                                          size_t image_size,
                                          size_t grid_size,
                                          float  conv_corr_norm_factor){
    const int support = 8;
    size_t t1_t2 = blockIdx.x*blockDim.x + threadIdx.x;
    if(t1_t2 < image_size / 2 + 1){
        float t1_t2_norm = (float)t1_t2 / grid_size;
        float correction = 0.0;
        for(int i=0; i < sizeof(quadrature_nodes)/sizeof(*quadrature_nodes); i++){
            float angle = t1_t2_norm * support * quadrature_nodes[i];
            correction += quadrature_kernel[i] * quadrature_weights[i] * cospif(angle);
        }
        conv_corr_kernel[t1_t2] = correction * support / conv_corr_norm_factor;
    }
}

__global__ void fip_pipe_cuda_gridding(cufftComplex*       grid,
                                       const cufftComplex* visibilities,
                                       const float*        coords,
                                       const float         transform[3][3],
                                       const size_t        grid_stride,
                                       const size_t        grid_size,
                                       const size_t        num_baselines,
                                       const float         r1r2_scale){
    const int     KERNEL_SUPPORT_BOUND = 16;
    const int     support              = 8;
    const int     half_support         = support / 2;
    const float   inv_half_support     = 1.0f / half_support;
    const float   beta                 = 15.3704324328;
    const float   weight               = fabsf(transform[0][0]*transform[1][1] -
                                               transform[0][1]*transform[1][0]);
    const size_t  idx                  = blockIdx.x * blockDim.x + threadIdx.x;
    const long    grid_size_l          = (long)grid_size;
    const long    grid_stride_l        = (long)grid_stride;
    const long    grid_min_r1r2        = -grid_size_l      / 2;
    const long    grid_max_r1r2        = (grid_size_l - 1) / 2;
    const size_t  grid_size_half       =  grid_size        / 2;
    cufftComplex* grid_origin          = &grid[grid_stride*grid_size_half +
                                                           grid_size_half];
    cufftComplex  v;
    float         k;
    float         r1_kernel[KERNEL_SUPPORT_BOUND];
    float         r2_kernel[KERNEL_SUPPORT_BOUND];
    long          r1, r2;


    if(idx < num_baselines){
        const float r1_pos = coords[idx*2+0] * r1r2_scale;
        const float r2_pos = coords[idx*2+1] * r1r2_scale;
        const long  r1_min = max((long)ceilf (r1_pos - half_support), grid_min_r1r2);
        const long  r1_max = min((long)floorf(r1_pos + half_support), grid_max_r1r2);
        const long  r2_min = max((long)ceilf (r2_pos - half_support), grid_min_r1r2);
        const long  r2_max = min((long)floorf(r2_pos + half_support), grid_max_r1r2);

        if(r1_min > r1_max ||
           r2_min > r2_max)
            return;

        for(r1=r1_min; r1<=r1_max; r1++)
            r1_kernel[r1 - r1_min] = exp_semicircle(beta, (r1-r1_pos) * inv_half_support);
        for(r2=r2_min; r2<=r2_max; r2++)
            r2_kernel[r2 - r2_min] = exp_semicircle(beta, (r2-r2_pos) * inv_half_support);

        for(r1=r1_min; r1<=r1_max; r1++){
            for(r2=r2_min; r2<=r2_max; r2++){
                k = r1_kernel[r1-r1_min] *
                    r2_kernel[r2-r2_min];
                if((r1+r2) & 1)
                    k = -k;

                v = visibilities[idx];
                atomicAdd(&grid_origin[r1*grid_stride_l + r2].x, (v.x/weight) * k);
                atomicAdd(&grid_origin[r1*grid_stride_l + r2].y, (v.y/weight) * k);
            }
        }
    }
}

__global__ void fip_pipe_cuda_interp  (float*              image,
                                       const cufftComplex* grid,
                                       const float         transform[3][3],
                                       const size_t        image_stride,
                                       const size_t        image_size,
                                       const size_t        grid_stride,
                                       const size_t        grid_size,
                                       const float         dc_rad,
                                       const float*        conv_corr_kernel,
                                       const float         conv_corr_norm_factor,
                                       const float         inv_num_baselines){
    const long          image_stride_l    =  image_stride;
    const size_t        image_size_half   =  image_size/2;
    const long          image_size_half_l =  image_size_half;
    float*              image_origin      = &image[image_size_half*image_stride + image_size_half];
    const long          grid_stride_l     =  grid_stride;
    const size_t        grid_size_half    =  grid_size/2;
    const cufftComplex* grid_origin       = &grid[grid_size_half  *grid_stride  + grid_size_half];

    const size_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy  = blockIdx.y * blockDim.y + threadIdx.y;
    const float  V00  = transform[0][0];
    const float  V01  = transform[0][1];
    const float  V10  = transform[1][0];
    const float  V11  = transform[1][1];
    const float  V20  = transform[2][0];
    const float  V21  = transform[2][1];
    const float  V22  = transform[2][2];
    const float  di2  = image_size*0.5f;
    const float  idxf =       idx - di2;               /* Reduced by half image size. Float */
    const float  idyf =       idy - di2;               /* Reduced by half image size. Float */
    const long   idxr = (long)idx - image_size_half_l; /* Reduced by half image size. Integer */
    const long   idyr = (long)idy - image_size_half_l; /* Reduced by half image size. Integer */
    const float  r0   = 180.0f / M_PI;
    const float  dc   = dc_rad / M_PI * 180;
    const float  xi   = V20/V22;
    const float  eta  = V21/V22;

    float        oi0, oi1;                           /* (ex-) output_index[k+0], output_index[k+1] */
    float        pixel_sum;                          /* (ex-) dirty_pre[idy*di + idx] */

    if(idx<image_size && idy<image_size){
        /**
         * Kernel (ex-)coordschange().
         *
         * Because of fusion, the following no longer needs to be spilled and
         * reloaded from memory:
         *
         * p1 = output_index[(idx*di+idy)*2+0]
         *    = ( -V[0][0]*(idx-di2) + V[1][0]*(idy-di2) ) / fabs(V[2][2]) + di2
         * p2 = output_index[(idx*di+idy)*2+1]
         *    = ( -V[0][1]*(idx-di2) + V[1][1]*(idy-di2) ) / fabs(V[2][2]) + di2
         */

        const float p1 = (-V00*idxf + V10*idyf) / fabsf(V22) + di2;
        const float p2 = (-V01*idxf + V11*idyf) / fabsf(V22) + di2;


        /**
         * Kernel (ex-)p2p().
         *
         * Because of fusion, the following no longer needs to be spilled and
         * reloaded from memory:
         *
         * oi0 = output_index[(idx*di+idy)*2+0]
         * oi1 = output_index[(idx*di+idy)*2+1]
         *
         * According to paper:
         *     M. R.  Calabretta, E. W.  Greisen, 'Representations of celestial coordinates in FITS,' A&A,395(3),1077-1122,2002.
         */

        float x   = -dc * (p1 - (di2 + 1.0f));
        float y   =  dc * (p2 - (di2 + 1.0f));
        float h   = hypotf(x, y);
        float hr0 = h/r0;

        float r, w, z;
        if(h != 0.0f){
            /**
             * Optimize sincosf(atan2f(x, -y), &x, &y) into x/=h, y/=-h.
             *
             * Example inputs and comparisons:
             *
             *    Input   | atan2f(x, -y) | sincosf(atan2f(x, -y), &x, &y) |  x/=h, y/=-h
             * -----------+---------------+--------------------------------+--------------
             * x=1,  y=0  |      pi/2     |           x=1,  y=0            |  x=1,  y=0
             * x=0,  y=1  |      pi       |           x=0,  y=-1           |  x=0,  y=-1
             * x=-1, y=0  |    3*pi/2     |           x=-1, y=0            |  x=-1, y=0
             * x=0,  y=-1 |      0        |           x=0,  y=1            |  x=0,  y=1
             */

            x /=  h;
            y /= -h;
        }else{
            x  = 0.0f;
            y  = 1.0f;
        }

        /**
         * The original conditionals were
         *
         *     float x0 = x / r0;
         *     float y0 = y / r0;
         *     float r2 = x0 * x0 + y0 * y0;
         *
         *     if(r2 < 0.5f){
         *         A
         *     }else if(r2 <= 1.0f){
         *         B
         *     }else{
         *
         * Manipulating the equations,
         *
         *     r2 = x0 * x0 + y0 * y0
         *        = (x/r0)**2 + (y/r0)**2
         *        = (x**2 + y**2)   / r0**2
         *        = hypotf(x, y)**2 / r0**2
         *        = h**2 / r0**2
         *
         * we find the conditionals are equivalent to
         *
         *     if(h*h/r0/r0 < 0.5f){                  if(h*h < r0*r0*0.5f){                  if(h < r0*sqrtf(0.5f)){
         *         A                                      A                                      A
         *     }else if(h*h/r0/r0 <= 1.0f){    ==>    }else if(h*h <= r0*r0*1.0f){    ==>    }else if(h <= r0){
         *         B                                      B                                      B
         *     }else{                                 }else{                                 }else{
         */

        if(h <= r0){
            /**
             * Convert numerical expressions from the original into saner ones.
             *
             * An angle theta was originally calculated from one of two formulas,
             *
             *     theta = { acosf(sqrtf(r2))        ,        r2 <  0.5
             *             { asinf(sqrtf(1.0 - r2))  , 0.5 <= r2 <= 1.0
             *
             * Presumably for numerical reasons (sin^2 x = 1.0 - cos^2 x).
             * But the angle's sine and cosine were then immediately calculated.
             * That calls into question the utility of the foregoing.
             *
             * -------------------
             * COSTHE
             *
             *     Reformulate as follows:
             *
             *         costhe = cosf(theta)
             *                = cosf(acosf(sqrtf(r2)))
             *                = sqrtf(r2)
             *                = sqrtf(x0 * x0 + y0 * y0)
             *                = hypotf(x0, y0)
             *                = h/r0
             *
             *     As the only subsequent usage of costhe is
             *
             *              r = r0 * costhe
             *
             *     We may cancel even that usage:
             *
             *              r = r0 * costhe
             *                = r0 * h/r0
             *                = h
             *
             * -------------------
             * Z
             *
             *     z is immediately subtracted from 1.0. To preserve numerical stability,
             *     special handling should be undertaken knowing that downstream operation.
             *     We present the straightforward analysis and the one considering the 1.0-z
             *     subtraction:
             *
             *              z = sinf(theta)
             *                = sinf(asinf(sqrtf(1.0f - r2)))
             *                = sqrtf(1.0f - r2)
             *
             *       1.0f - z = 1.0f - sqrtf(1.0f - r2)
             *
             *     This is stable as r2 -> 1 because the result approaches 1, and unstable as
             *     r2 -> 0 because the result also approaches 0, but all precision is lost due
             *     to catastrophic cancellation. Thus, rewrite as follows:
             *
             *       1.0f - z =  1.0f - sqrtf(1.0f - r2)
             *                = (1.0f - sqrtf(1.0f - r2)) * (1.0f + sqrtf(1.0f - r2)) / (1.0f + sqrtf(1.0f - r2))
             *                = (1.0f - (sqrtf(1.0f - r2)))^2) / (1.0f + sqrtf(1.0f - r2))
             *                = (1.0f - (1.0f - r2)) / (1.0f + sqrtf(1.0f - r2))
             *                = r2 / (1.0f + sqrtf(1.0f - r2))
             *
             *     Let hr0 = h/r0, then r2 = hr0*hr0
             *
             *                = hr0*hr0 / (1.0f + sqrtf(1.0f - hr0*hr0))
             *
             *     which safely and accurately approaches 0 as hr0 -> 0 (equivalently, as h and r2 -> 0).
             */

            r = h;
            if(h < r0*sqrtf(0.5f)){
                z =            1.0f - sqrtf(1.0f - hr0*hr0);
            }else{
                z = hr0*hr0 / (1.0f + sqrtf(1.0f - hr0*hr0));
            }

            w = xi*xi + eta*eta;
            if(w == 0.0f){
                x =  r*x;
                y = -r*y;
            }else{
                x =  r*x + z*r0*xi;
                y = -r*y + z*r0*eta;
            }

            oi0 = -x/dc + di2 + 1.0f;
            oi1 =  y/dc + di2 + 1.0f;
        }else{
            /**
             * Because of the early skip here, we must spill to output_index the values
             * that *would* have been present by the legacy coordschange() had it actually
             * run to maintain perfect equivalence.
             *
             * Formerly:
             *
             *     output_index[(idx*di+idy)*2+0] = p1;
             *     output_index[(idx*di+idy)*2+1] = p2;
             *     return;
             */

            oi0 = p1;
            oi1 = p2;
        }


        /**
         * Kernel (ex-)accumulation().
         *
         * This kernel contains a deeply questionable sign-flipping of the pixels that is
         * probably the compensation of an ifftshift formerly in the codebase.
         *
         * Avoid spill and reload by not writing out to memory in this part of the fusion.
         */

        pixel_sum = grid_origin[grid_stride_l*idyr + idxr].x;
        if(idxr+idyr & 1){
            pixel_sum = - pixel_sum;
        }


        /**
         * Kernel (ex-)scaling().
         *
         * Avoid spill and reload by using pixel_sum directly from the registers.
         *
         * Because of fusion, the following no longer needs to be spilled and
         * reloaded from memory:
         *
         * dirty_pre[idy*di + idx] = fabs(pixel_sum);
         */

        pixel_sum *= 1 / (conv_corr_kernel[abs(idxr)] *
                          conv_corr_kernel[abs(idyr)] *
                          conv_corr_norm_factor       *
                          conv_corr_norm_factor);
        pixel_sum  = fabs(pixel_sum);


        /**
         * Kernel (ex-)finalinterp().
         *
         * Because of fusion, the following no longer needs to be spilled and
         * reloaded from memory:
         *
         * output_index[(idx*di+idy)*2+0] = oi0;
         * output_index[(idx*di+idy)*2+1] = oi1;
         * dirty_pre[idy*di + idx] = fabs(pixel_sum);
         */

        const float LL    = oi0 - image_size_half_l;
        const float MM    = oi1 - image_size_half_l;
        const float value = pixel_sum * inv_num_baselines;

        if(fabs(LL) < image_size_half_l-1 &&
           fabs(MM) < image_size_half_l-1){
            const float LLf  = floorf(LL);
            const float MMf  = floorf(MM);
            const float LLc  = ceilf (LL);/* Theoretically LLf+1 except if LL was integer */
            const float MMc  = ceilf (MM);/* Theoretically MMf+1 except if MM was integer */

            const long  LLfi = LLf;
            const long  LLci = LLc;
            const long  MMfi = MMf;
            const long  MMci = MMc;

            atomicAdd(&image_origin[image_stride_l*MMfi + LLfi],  (1-LL+LLf) * (1-MM+MMf) * value);/* Always effective                  */
            atomicAdd(&image_origin[image_stride_l*MMci + LLfi],  (1-LL+LLf) * (0+MM-MMf) * value);/* Ineffective when       MM integer */
            atomicAdd(&image_origin[image_stride_l*MMfi + LLci],  (0+LL-LLf) * (1-MM+MMf) * value);/* Ineffective when LL       integer */
            atomicAdd(&image_origin[image_stride_l*MMci + LLci],  (0+LL-LLf) * (0+MM-MMf) * value);/* Ineffective when LL or MM integer */
        }
    }
}

__global__ void fip_pipe_cuda_tlisi   (float*       result,
                                       const float* image0,
                                       const float* max0,
                                       const float* image1,
                                       const float* max1,
                                       const float* image2,
                                       const float* max2,
                                       const size_t result_stride,
                                       const size_t image_stride,
                                       const size_t image_size,
                                       const size_t unit_size,
                                       const size_t unit_num,
                                       const float  C){
    extern  __shared__  float sharedNumDen[];

    const float  maxallval = fmaxf(*max0, fmaxf(*max1, *max2));
    const size_t bid       = blockIdx.x; // tile index
    const size_t tid       = threadIdx.x;

    const size_t i_id      = bid / unit_num;
    const size_t j_id      = bid % unit_num;
    const size_t factor    = ceiling_divide(unit_size*unit_size, 1024);

    for(size_t f=0; f<factor; f++){
        if(tid+f*1024 < unit_size*unit_size){
            if(f == 0){
                sharedNumDen[tid+   0] = 0; /* Sum of diff_out */
                sharedNumDen[tid+1024] = 0; /* Max of diff_out */
                sharedNumDen[tid+2048] = 0; /* Sum of r        */
            }
            const size_t rows = (tid + f*1024) / unit_size;
            const size_t cols = (tid + f*1024) % unit_size;

            const size_t I_id = i_id * unit_size + rows;
            const size_t J_id = j_id * unit_size + cols;
            const size_t off  = I_id * image_stride + J_id;

            const float  img_val0 = image0[off],
                         img_val1 = image1[off],
                         img_val2 = image2[off],
                         abs_df01 = fabsf(img_val0-img_val1),
                         abs_df12 = fabsf(img_val1-img_val2),
                         diff_out = fabsf(abs_df01-abs_df12),
                         snap_val = img_val1<=0 ? C : img_val1;

            sharedNumDen[tid+   0] =                             sharedNumDen[tid+   0] + diff_out;
            sharedNumDen[tid+1024] =                         max(sharedNumDen[tid+1024],  diff_out);
            sharedNumDen[tid+2048] = (diff_out / snap_val < 1) ? sharedNumDen[tid+2048] + diff_out / snap_val :
                                                                 sharedNumDen[tid+2048] + 1;
        }else{
            if(f == 0){
                sharedNumDen[tid+   0] = 0; /* Sum of diff_out */
                sharedNumDen[tid+1024] = 0; /* Max of diff_out */
                sharedNumDen[tid+2048] = 0; /* Sum of r        */
            }
        }
    }

    for(size_t d = blockDim.x/2; d>0; d/=2){
        __syncthreads();
        if(tid<d){
            sharedNumDen[tid+   0] +=     sharedNumDen[tid+d];
            sharedNumDen[tid+1024]  = max(sharedNumDen[tid+1024],
                                          sharedNumDen[tid+1024+d]);
            sharedNumDen[tid+2048] +=     sharedNumDen[tid+2048+d];
        }
    }

    if(tid==0){
        result[i_id*result_stride + j_id] =
            1 - (sharedNumDen[0   ]/unit_size/unit_size) *
                 sharedNumDen[1024]                      *
                (sharedNumDen[2048]/unit_size/unit_size) / maxallval / maxallval;
    }
}



/*************************************************************************/
static cudaError     fip_pipe_cuda_select_device             (fip_pipe_cuda_state*     pipe){
    cudaError_t cudaError;

    if((cudaError = cudaGetDevice(&pipe->device.ordinal))){
        fprintf(stderr, "Cannot find CUDA device: %s (%d)\n",
                cudaGetErrorString(cudaError),
                (int)cudaError);
        return cudaError;
    }

    if((cudaError = cudaGetDeviceProperties(&pipe->device.props,
                                             pipe->device.ordinal))){
        fprintf(stderr, "Cannot get the properties of CUDA device with ordinal %d: %s (%d)\n",
                        (int)pipe->device.ordinal,
                        cudaGetErrorString(cudaError),
                        (int)cudaError);
        return cudaError;
    }else{
        if(pipe->device.props.pciDomainID != 0){
            snprintf(pipe->device.pci, sizeof(pipe->device.pci),
                     "%04x:%02x:%02x.0",
                     pipe->device.props.pciDomainID,
                     pipe->device.props.pciBusID,
                     pipe->device.props.pciDeviceID);
        }else{
            snprintf(pipe->device.pci, sizeof(pipe->device.pci),
                     "%02x:%02x.0",
                     pipe->device.props.pciBusID,
                     pipe->device.props.pciDeviceID);
        }
        snprintf(pipe->device.name, sizeof(pipe->device.name), "%s",
                 pipe->device.props.name);
        snprintf(pipe->device.uuid, sizeof(pipe->device.uuid),
                 "GPU-%02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-"
                     "%02hhx%02hhx-%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx",
                 pipe->device.props.uuid.bytes[ 0], pipe->device.props.uuid.bytes[ 1],
                 pipe->device.props.uuid.bytes[ 2], pipe->device.props.uuid.bytes[ 3],
                 pipe->device.props.uuid.bytes[ 4], pipe->device.props.uuid.bytes[ 5],
                 pipe->device.props.uuid.bytes[ 6], pipe->device.props.uuid.bytes[ 7],
                 pipe->device.props.uuid.bytes[ 8], pipe->device.props.uuid.bytes[ 9],
                 pipe->device.props.uuid.bytes[10], pipe->device.props.uuid.bytes[11],
                 pipe->device.props.uuid.bytes[12], pipe->device.props.uuid.bytes[13],
                 pipe->device.props.uuid.bytes[14], pipe->device.props.uuid.bytes[15]);
    }

    if(pipe->device.props.maxThreadsPerBlock < 1024){
        fprintf(stderr, "Selected CUDA device supports fewer than 1024 threads/block! (%d)\n",
                        pipe->device.props.maxThreadsPerBlock);
        return cudaErrorInvalidValue;
    }

#if 1
    if(1){
        printf("Selected GPU %d: %s (UUID: %s, PCIe %s)\n",
               pipe->device.ordinal,
               pipe->device.name,
               pipe->device.uuid,
               pipe->device.pci);
    }
#endif

    return cudaSuccess;
}

static cudaError     fip_pipe_cuda_plan_mem                  (fip_pipe_cuda_state*     pipe){
    cudaError_t cudaError;
    size_t      memfree=0, memtotal=0, memest=0;

    pipe->ring.depth.vis_bin   = 2;
    pipe->ring.depth.transform = 2;
    pipe->ring.depth.grid      = 3;
    pipe->ring.depth.image     = 5;
    pipe->ring.depth.result    = 2;

    if((cudaError = cudaMemGetInfo(&memfree, &memtotal))){
        fprintf(stderr, "Cannot query free memory on selected device! %s (%d)\n",
                        cudaGetErrorString(cudaError), (int)cudaError);
        fflush (stderr);
        return cudaError;
    }

    memest = pipe->ring.depth.vis_bin   * pipe->param.num_baselines     * sizeof(cuComplex) +  /* Visibilities */
             pipe->ring.depth.vis_bin   * pipe->param.num_baselines * 2 * sizeof(float)     +  /* Coordinates */
             pipe->ring.depth.transform * 3                         * 3 * sizeof(float)     +  /* Transform */
             pipe->ring.depth.grid      * pipe->param.grid_size     *
                                          pipe->param.grid_size     *     sizeof(cuComplex) +  /* Grid */
             pipe->ring.depth.image     * pipe->param.image_size    *
                                          pipe->param.image_size    *     sizeof(float)     +  /* Image */
             pipe->ring.depth.result    * pipe->param.unit_num      *
                                          pipe->param.unit_num      *     sizeof(float)     +  /* Result */
             pipe->ring.depth.image                                 *     sizeof(float);       /* Max */

    if(memest > memtotal){
        fprintf(stderr, "GPU %d (%s) too small!\n"
                        "A minimum %zu bytes of memory are required, but device only has %zu bytes of memory total!\n",
                        pipe->device.ordinal,
                        pipe->device.name,
                        memest,
                        memtotal);
        fflush (stderr);
        return cudaErrorMemoryAllocation;
    }

    if(memest > memfree){
        fprintf(stderr, "Insufficient free memory on GPU %d (%s)!\n"
                        "A minimum %zu bytes of memory are required, but only %zu bytes of memory free!\n"
                        "Is another application using the GPU?\n",
                        pipe->device.ordinal,
                        pipe->device.name,
                        memest,
                        memfree);
        fflush (stderr);
        return cudaErrorMemoryAllocation;
    }

    return cudaSuccess;
}

static cudaError     fip_pipe_cuda_plan_launch               (fip_pipe_cuda_state*     pipe){
    /**
     * There are at least four typical CUDA kernel launch configurations:
     *
     *   NAME            #THRD  #BLOCK                             SHMEM
     *   "s" (Square):   32x32, ~image_size/32 x ~image_size/32
     *   "k" (Convolve): 1024,  ~(image_size/2+1)/1024
     *   "g" (Gridding): 1024,  ~num_baselines/1024
     *   "t" (TLISI):    1024,   unit_num*unit_num                 3*1024 floats
     *
     * Abbreviate them and centralize their calculations here.
     */

    pipe->launch.Ts.x = 32;
    pipe->launch.Ts.y = 32;
    pipe->launch.Ts.z = 1;
    pipe->launch.Bs.x = (unsigned)ceiling_divide(pipe->param.image_size,     pipe->launch.Ts.x);
    pipe->launch.Bs.y = (unsigned)ceiling_divide(pipe->param.image_size,     pipe->launch.Ts.y);
    pipe->launch.Bs.z = 1;

    pipe->launch.Tk.x = 1024;
    pipe->launch.Tk.y = 1;
    pipe->launch.Tk.z = 1;
    pipe->launch.Bk.x = (unsigned)ceiling_divide(pipe->param.image_size/2+1, pipe->launch.Tk.x);
    pipe->launch.Bk.y = 1;
    pipe->launch.Bk.z = 1;

    pipe->launch.Tg.x = 1024;
    pipe->launch.Tg.y = 1;
    pipe->launch.Tg.z = 1;
    pipe->launch.Bg.x = (unsigned)ceiling_divide(pipe->param.num_baselines,  pipe->launch.Tg.x);
    pipe->launch.Bg.y = 1;
    pipe->launch.Bg.z = 1;

    pipe->launch.Tt.x = 1024;
    pipe->launch.Tt.y = 1;
    pipe->launch.Tt.z = 1;
    pipe->launch.Bt.x = (unsigned)(pipe->param.unit_num * pipe->param.unit_num);
    pipe->launch.Bt.y = 1;
    pipe->launch.Bt.z = 1;
    pipe->launch.St   = 3 * pipe->launch.Tt.x * sizeof(float);

    return cudaSuccess;
}

static cudaError     fip_pipe_cuda_destroy_events            (fip_pipe_cuda_state*     pipe){
    size_t t, i;

    cudaEventDestroy(pipe->events.pipestart);
    cudaEventDestroy(pipe->events.loopstart);
    cudaEventDestroy(pipe->events.loopend);
    cudaEventDestroy(pipe->events.pipeend);

    for(t=0; t<sizeof(pipe->events.iter) /
               sizeof(pipe->events.iter[0]); t++){
        for(i=ITER_START; i<ITER_END; i++){
            cudaEventDestroy(pipe->events.iter[t][i]);
        }
    }

    for(i=0; i<pipe->ring.depth.vis_bin; i++)
        cudaEventDestroy(pipe->events.ring.vis_pinned[i]);
    for(i=0; i<pipe->ring.depth.vis_bin; i++)
        cudaEventDestroy(pipe->events.ring.vis_gpu[i]);
    for(i=0; i<pipe->ring.depth.grid; i++)
        cudaEventDestroy(pipe->events.ring.grid_gpu[i]);
    for(i=0; i<pipe->ring.depth.image; i++)
        cudaEventDestroy(pipe->events.ring.image_gpu[i]);
    for(i=0; i<pipe->ring.depth.result; i++)
        cudaEventDestroy(pipe->events.ring.result_gpu[i]);
    for(i=0; i<pipe->ring.depth.result; i++)
        cudaEventDestroy(pipe->events.ring.result_pinned[i]);

    free(pipe->events.ring.vis_pinned);
    free(pipe->events.ring.vis_gpu);
    free(pipe->events.ring.grid_gpu);
    free(pipe->events.ring.image_gpu);
    free(pipe->events.ring.result_gpu);
    free(pipe->events.ring.result_pinned);

    pipe->events.ring.vis_pinned    = NULL;
    pipe->events.ring.vis_gpu       = NULL;
    pipe->events.ring.grid_gpu      = NULL;
    pipe->events.ring.image_gpu     = NULL;
    pipe->events.ring.result_gpu    = NULL;
    pipe->events.ring.result_pinned = NULL;

    return cudaSuccess;
}

static cudaError     fip_pipe_cuda_create_events             (fip_pipe_cuda_state*     pipe){
    size_t t, i;

    pipe->events.ring.vis_pinned    = (cudaEvent_t*)calloc(pipe->ring.depth.vis_bin, sizeof(cudaEvent_t));
    pipe->events.ring.vis_gpu       = (cudaEvent_t*)calloc(pipe->ring.depth.vis_bin, sizeof(cudaEvent_t));
    pipe->events.ring.grid_gpu      = (cudaEvent_t*)calloc(pipe->ring.depth.grid,    sizeof(cudaEvent_t));
    pipe->events.ring.image_gpu     = (cudaEvent_t*)calloc(pipe->ring.depth.image,   sizeof(cudaEvent_t));
    pipe->events.ring.result_gpu    = (cudaEvent_t*)calloc(pipe->ring.depth.result,  sizeof(cudaEvent_t));
    pipe->events.ring.result_pinned = (cudaEvent_t*)calloc(pipe->ring.depth.result,  sizeof(cudaEvent_t));

    if(!pipe->events.ring.vis_pinned   ||
       !pipe->events.ring.vis_gpu      ||
       !pipe->events.ring.grid_gpu     ||
       !pipe->events.ring.image_gpu    ||
       !pipe->events.ring.result_gpu   ||
       !pipe->events.ring.result_pinned){
        free(pipe->events.ring.vis_pinned);
        free(pipe->events.ring.vis_gpu);
        free(pipe->events.ring.grid_gpu);
        free(pipe->events.ring.image_gpu);
        free(pipe->events.ring.result_gpu);
        free(pipe->events.ring.result_pinned);

        pipe->events.ring.vis_pinned    = NULL;
        pipe->events.ring.vis_gpu       = NULL;
        pipe->events.ring.grid_gpu      = NULL;
        pipe->events.ring.image_gpu     = NULL;
        pipe->events.ring.result_gpu    = NULL;
        pipe->events.ring.result_pinned = NULL;

        return cudaErrorMemoryAllocation;
    }

    cudaEventCreate(&pipe->events.pipestart);
    cudaEventCreate(&pipe->events.loopstart);
    cudaEventCreate(&pipe->events.loopend);
    cudaEventCreate(&pipe->events.pipeend);

    for(t=0; t<sizeof(pipe->events.iter) /
               sizeof(pipe->events.iter[0]); t++){
        for(i=ITER_START; i<ITER_END; i++){
            cudaEventCreate(&pipe->events.iter[t][i]);
        }
    }

    for(i=0; i<pipe->ring.depth.vis_bin; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.vis_pinned[i],    cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.vis_pinned[i]);
    }
    for(i=0; i<pipe->ring.depth.vis_bin; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.vis_gpu[i],       cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.vis_gpu[i]);
    }
    for(i=0; i<pipe->ring.depth.grid; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.grid_gpu[i],      cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.grid_gpu[i]);
    }
    for(i=0; i<pipe->ring.depth.image; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.image_gpu[i],     cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.image_gpu[i]);
    }
    for(i=0; i<pipe->ring.depth.result; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.result_gpu[i],    cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.result_gpu[i]);
    }
    for(i=0; i<pipe->ring.depth.result; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.result_pinned[i], cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.result_pinned[i]);
    }

    return cudaSuccess;
}

static cudaError     fip_pipe_cuda_record                    (fip_pipe_cuda_state*     pipe,
                                                              size_t                   iter,
                                                              fip_pipe_cuda_event      code,
                                                              cudaStream_t             stream,
                                                              int                      flags){
    switch(code){
        case PIPE_START: return cudaEventRecordWithFlags(pipe->events.pipestart, stream, flags);
        case LOOP_START: return cudaEventRecordWithFlags(pipe->events.loopstart, stream, flags);
        case PIPE_END:   return cudaEventRecordWithFlags(pipe->events.pipeend,   stream, flags);
        case LOOP_END:   return cudaEventRecordWithFlags(pipe->events.loopend,   stream, flags);
        default:
            iter %= sizeof(pipe->events.iter) /
                    sizeof(pipe->events.iter[0]);
            return cudaEventRecordWithFlags(pipe->events.iter[iter][code], stream, flags);
    }
}

static cudaError     fip_pipe_cuda_await                     (fip_pipe_cuda_state*     pipe,
                                                              size_t                   iter,
                                                              fip_pipe_cuda_event      code,
                                                              cudaStream_t             stream,
                                                              int                      flags){
    switch(code){
        case PIPE_START: return cudaStreamWaitEvent(stream, pipe->events.pipestart, flags);
        case LOOP_START: return cudaStreamWaitEvent(stream, pipe->events.loopstart, flags);
        case PIPE_END:   return cudaStreamWaitEvent(stream, pipe->events.pipeend,   flags);
        case LOOP_END:   return cudaStreamWaitEvent(stream, pipe->events.loopend,   flags);
        default:
            iter %= sizeof(pipe->events.iter) /
                    sizeof(pipe->events.iter[0]);
            return cudaStreamWaitEvent(stream, pipe->events.iter[iter][code], flags);
    }
}

static cudaError     fip_pipe_cuda_lock                      (fip_pipe_cuda_state*     pipe,
                                                              size_t                   iter,
                                                              fip_pipe_cuda_ring       ring,
                                                              cudaStream_t             stream,
                                                              int                      flags){
    switch(ring){
        case RING_VIS_PIN:
            iter %= pipe->ring.depth.vis_bin;
            return cudaStreamWaitEvent(stream, pipe->events.ring.vis_pinned[iter],    flags);
        case RING_VIS_GPU:
            iter %= pipe->ring.depth.vis_bin;
            return cudaStreamWaitEvent(stream, pipe->events.ring.vis_gpu[iter],       flags);
        case RING_GRID_GPU:
            iter %= pipe->ring.depth.grid;
            return cudaStreamWaitEvent(stream, pipe->events.ring.grid_gpu[iter],      flags);
        case RING_IMAGE_GPU:
            iter %= pipe->ring.depth.image;
            return cudaStreamWaitEvent(stream, pipe->events.ring.image_gpu[iter],     flags);
        case RING_RESULT_GPU:
            iter %= pipe->ring.depth.result;
            return cudaStreamWaitEvent(stream, pipe->events.ring.result_gpu[iter],    flags);
        case RING_RESULT_PIN:
            iter %= pipe->ring.depth.result;
            return cudaStreamWaitEvent(stream, pipe->events.ring.result_pinned[iter], flags);
        default:
            return cudaErrorInvalidValue;
    }
}

static cudaError     fip_pipe_cuda_unlock                    (fip_pipe_cuda_state*     pipe,
                                                              size_t                   iter,
                                                              fip_pipe_cuda_ring       ring,
                                                              cudaStream_t             stream,
                                                              int                      flags){
    switch(ring){
        case RING_VIS_PIN:
            iter %= pipe->ring.depth.vis_bin;
            return cudaEventRecordWithFlags(pipe->events.ring.vis_pinned[iter],    stream, flags);
        case RING_VIS_GPU:
            iter %= pipe->ring.depth.vis_bin;
            return cudaEventRecordWithFlags(pipe->events.ring.vis_gpu[iter],       stream, flags);
        case RING_GRID_GPU:
            iter %= pipe->ring.depth.grid;
            return cudaEventRecordWithFlags(pipe->events.ring.grid_gpu[iter],      stream, flags);
        case RING_IMAGE_GPU:
            iter %= pipe->ring.depth.image;
            return cudaEventRecordWithFlags(pipe->events.ring.image_gpu[iter],     stream, flags);
        case RING_RESULT_GPU:
            iter %= pipe->ring.depth.result;
            return cudaEventRecordWithFlags(pipe->events.ring.result_gpu[iter],    stream, flags);
        case RING_RESULT_PIN:
            iter %= pipe->ring.depth.result;
            return cudaEventRecordWithFlags(pipe->events.ring.result_pinned[iter], stream, flags);
        default:
            return cudaErrorInvalidValue;
    }
}

static cudaError     fip_pipe_cuda_plan_npp                  (fip_pipe_cuda_state*     pipe,
                                                              NppiSize*                npp_image_size,
                                                              NppStreamContext*        npp_ctx){
    npp_image_size->height = (int)pipe->param.image_size;
    npp_image_size->width  = (int)pipe->param.image_size;
    nppGetStreamContext(npp_ctx);
    npp_ctx->hStream       = pipe->stream.gridding;
    cudaStreamGetFlags(npp_ctx->hStream, &npp_ctx->nStreamFlags);
    nppiMaxGetBufferHostSize_32f_C1R_Ctx(*npp_image_size, &pipe->wrkspc.npp.sz, *npp_ctx);
    return cudaSuccess;
}

static cufftResult   fip_pipe_cuda_plan_fft                  (fip_pipe_cuda_state*     pipe,
                                                              cufftHandle*             plan){
    cufftResult cufftError;

    /**
     * cuFFT Advanced Data Layout (ADL) Parameters.
     *
     * These are not cited where they should be [1], but quoting [2]:
     *
     *     Advanced parameters are defined in units of the relevant data type
     *     (cufftReal, cufftDoubleReal, cufftComplex, or cufftDoubleComplex).
     *
     *     Advanced layout can be perceived as an additional layer of abstraction
     *     above the access to input/output data arrays. An element of coordinates
     *     [z][y][x] in signal number b in the batch will be associated with the
     *     following addresses in the memory:
     *
     *         1D
     *              input [b * idist + x * istride]
     *              output[b * odist + x * ostride]
     *
     *         2D
     *              input [b * idist +  (x * inembed[1] + y) * istride]
     *              output[b * odist +  (x * onembed[1] + y) * ostride]
     *
     *         3D
     *              input [b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
     *              output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]
     *
     *
     * [1]: https://docs.nvidia.com/cuda/cufft/index.html#c.cufftPlanMany
     * [2]: https://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout
     */

    int n[2]       = {(int)pipe->param.grid_size,
                      (int)pipe->param.grid_size};
    int strides[2] = {0,
                      (int)pipe->ring.stride.grid};

    if((cufftError = cufftCreate(plan))){
        fprintf(stderr, "Cannot create cuFFT plan! (%d)\n", (int)cufftError);
        return cufftError;
    }
    if((cufftError = cufftSetStream(*plan, pipe->stream.fft))){
        fprintf(stderr, "Cannot assign stream to cuFFT plan! (%d)\n", (int)cufftError);
        return cufftError;
    }
    if((cufftError = cufftPlanMany(plan, 2, n,
                                   strides, 1, pipe->ring.stride.grid*pipe->param.grid_size,
                                   strides, 1, pipe->ring.stride.grid*pipe->param.grid_size,
                                   CUFFT_C2C, 1))){
        fprintf(stderr, "Cannot make cuFFT plan for grid of size %zu (%d)\n",
                        pipe->param.grid_size, (int)cufftError);
        return cufftError;
    }
    return CUFFT_SUCCESS;
}

static cudaError     fip_pipe_cuda_free_mem                  (fip_pipe_cuda_state*     pipe){
    cudaFree    (pipe->ring.vis_bin_gpu);
    cudaFree    (pipe->ring.transform_gpu);
    cudaFree    (pipe->ring.grid_gpu);
    cudaFree    (pipe->ring.image_gpu);
    cudaFree    (pipe->ring.result_gpu);

    cudaFree    (pipe->wrkspc.npp.ptr);
    cudaFree    (pipe->wrkspc.cufft.ptr);
    cudaFree    (pipe->wrkspc.conv_corr_kernel);
    cudaFree    (pipe->ring.max_gpu);

    cudaFreeHost(pipe->ring.vis_bin_pinned);
    cudaFreeHost(pipe->ring.transform_pinned);
    cudaFreeHost(pipe->ring.result_pinned);

    return cudaSuccess;
}

static cudaError     fip_pipe_cuda_alloc_mem                 (fip_pipe_cuda_state*     pipe){
    cudaError_t cudaError;

    size_t num_baselines       = pipe->param.num_baselines;
    size_t grid_size           = pipe->param.grid_size;
    size_t image_size          = pipe->param.image_size;
    size_t unit_num            = pipe->param.unit_num;

    cudaMallocPitch(&pipe->ring.vis_bin_gpu,
                    &pipe->ring.stride.vis_bin,
                    num_baselines * sizeof(float),
                    pipe->ring.depth.vis_bin * (2+2));
    cudaMallocPitch(&pipe->ring.transform_gpu,
                    &pipe->ring.stride.transform,
                    3 * 3         * sizeof(float),
                    pipe->ring.depth.transform);
    cudaMallocPitch(&pipe->ring.grid_gpu,
                    &pipe->ring.stride.grid,
                    grid_size     * sizeof(cuComplex),
                    pipe->ring.depth.grid  * grid_size);
    cudaMallocPitch(&pipe->ring.image_gpu,
                    &pipe->ring.stride.image,
                    image_size    * sizeof(float),
                    pipe->ring.depth.image * image_size);
    cudaMallocPitch(&pipe->ring.result_gpu,
                    &pipe->ring.stride.result,
                    unit_num      * sizeof(float),
                    pipe->ring.depth.result * unit_num);

    pipe->ring.stride.vis_bin    /= sizeof(float);
    pipe->ring.stride.transform  /= sizeof(float);
    pipe->ring.stride.grid       /= sizeof(cuComplex);
    pipe->ring.stride.image      /= sizeof(float);
    pipe->ring.stride.result     /= sizeof(float);

    cudaMalloc     (&pipe->wrkspc.npp.ptr,              pipe->wrkspc.npp.sz);
    cudaMalloc     (&pipe->wrkspc.cufft.ptr,            pipe->wrkspc.cufft.sz);
    cudaMalloc     (&pipe->wrkspc.conv_corr_kernel,     (image_size/2+1)                                 * sizeof(float));
    cudaMalloc     (&pipe->ring.max_gpu,                pipe->ring.depth.image                           * sizeof(float));

    cudaMallocHost (&pipe->ring.vis_bin_pinned,         pipe->ring.depth.vis_bin * num_baselines * (2+2) * sizeof(float));
    cudaMallocHost (&pipe->ring.transform_pinned,       pipe->ring.depth.transform *       3 *         3 * sizeof(float));
    cudaMallocHost (&pipe->ring.result_pinned,          pipe->ring.depth.result  * unit_num  *  unit_num * sizeof(float));

    if((cudaError = cudaGetLastError())){
        fprintf(stderr, "ERROR! Failed to allocate memmory.\n"
                        "CUDA error code: %d; string: %s;\n",
                        (int)cudaError,
                        cudaGetErrorString(cudaError));
        fip_pipe_cuda_free_mem(pipe);
    }

    return cudaError;
}

static float*        fip_pipe_cuda_calc_ring_vis_pinned      (fip_pipe_cuda_state*     pipe, size_t i){
    i %= pipe->ring.depth.vis_bin;
    return (float*)pipe->ring.vis_bin_pinned +
                   pipe->param.num_baselines * (2+2) * i;
}

static cufftComplex* fip_pipe_cuda_calc_ring_vis_gpu         (fip_pipe_cuda_state*     pipe, size_t i){
    i %= pipe->ring.depth.vis_bin;
    return (cufftComplex*)((float*)pipe->ring.vis_bin_gpu    +
                                   pipe->ring.stride.vis_bin * (i*(2+2) + 0));
}

static float*        fip_pipe_cuda_calc_ring_coords_pinned   (fip_pipe_cuda_state*     pipe, size_t i){
    return (float*)fip_pipe_cuda_calc_ring_vis_pinned(pipe, i) + pipe->param.num_baselines*2;
}

static float*        fip_pipe_cuda_calc_ring_coords_gpu      (fip_pipe_cuda_state*     pipe, size_t i){
    return (float*)fip_pipe_cuda_calc_ring_vis_gpu(pipe, i) + pipe->ring.stride.vis_bin*2;
}

static float       (*fip_pipe_cuda_calc_ring_transform_pinned(fip_pipe_cuda_state*     pipe, size_t i))[3]{
    i %= pipe->ring.depth.transform;
    return (float(*)[3])((float*)pipe->ring.transform_pinned + 3*3*i);
}

static float       (*fip_pipe_cuda_calc_ring_transform_gpu   (fip_pipe_cuda_state*     pipe, size_t i))[3]{
    i %= pipe->ring.depth.transform;
    return (float(*)[3])((float*)pipe->ring.transform_gpu +
                                 pipe->ring.stride.transform * i);
}

static cufftComplex* fip_pipe_cuda_calc_ring_grid_gpu        (fip_pipe_cuda_state*     pipe, size_t i){
    i %= pipe->ring.depth.grid;
    return (cufftComplex*)pipe->ring.grid_gpu    +
                          pipe->param.grid_size  *
                          pipe->ring.stride.grid * i;
}

static float*        fip_pipe_cuda_calc_ring_max_gpu         (fip_pipe_cuda_state*     pipe, size_t i){
    i %= pipe->ring.depth.image;
    return (float*)pipe->ring.max_gpu + i;
}

static float*        fip_pipe_cuda_calc_ring_image_gpu       (fip_pipe_cuda_state*     pipe, size_t i){
    i %= pipe->ring.depth.image;
    return (float*)pipe->ring.image_gpu    +
                   pipe->param.image_size  *
                   pipe->ring.stride.image * i;
}

static float*        fip_pipe_cuda_calc_ring_result_gpu      (fip_pipe_cuda_state*     pipe, size_t i){
    i %= pipe->ring.depth.result;
    return (float*)pipe->ring.result_gpu    +
                   pipe->param.unit_num     *
                   pipe->ring.stride.result * i;
}

static float*        fip_pipe_cuda_calc_ring_result_pinned   (fip_pipe_cuda_state*     pipe, size_t i){
    i %= pipe->ring.depth.result;
    return (float*)pipe->ring.result_pinned +
                   pipe->param.unit_num     *
                   pipe->param.unit_num     * i;
}


typedef struct{
    fip_pipe_cuda_state*    pipe;
    void*                   userdata0;
    void*                   userdata1;
    fip_pipe_cuda_input_cb  input_cb;
    fip_pipe_cuda_output_cb output_cb;
    size_t                  input_iter;
    size_t                  output_iter;
} fipe_pipe_cuda_callback_state;

static void          fip_pipe_cuda_stage_data_read           (void* const p){
    fipe_pipe_cuda_callback_state* state = (fipe_pipe_cuda_callback_state*)p;
    fip_pipe_cuda_state*           pipe  = state->pipe;
    size_t                         i     = state->input_iter;

    state->input_cb (state->userdata0,
                     state->userdata1,
                     fip_pipe_cuda_calc_ring_vis_pinned      (pipe, i),
                     fip_pipe_cuda_calc_ring_coords_pinned   (pipe, i),
                     fip_pipe_cuda_calc_ring_transform_pinned(pipe, i),
                     pipe->param.num_baselines,
                     i);

    state->input_iter++;
}

static void          fip_pipe_cuda_stage_data_write          (void* const p){
    fipe_pipe_cuda_callback_state* state = (fipe_pipe_cuda_callback_state*)p;
    fip_pipe_cuda_state*           pipe  = state->pipe;
    size_t                         i     = state->output_iter;

    state->output_cb(state->userdata0,
                     state->userdata1,
                     fip_pipe_cuda_calc_ring_result_pinned   (pipe, i),
                     pipe->param.unit_num,
                     i);

    state->output_iter++;
}

int                  fip_pipe_cuda                           (fip_pipe_cuda_state*     pipe,
                                                              fip_pipe_cuda_input_cb   callback_input,
                                                              fip_pipe_cuda_output_cb  callback_output,
                                                              void*                    userdata0,
                                                              void*                    userdata1,
                                                              const size_t             snap_start,
                                                              const size_t             snap_end){
    const float      conv_corr_norm_factor = 2.4937047051153827;
    const float      C                     = 1e-6;
    const float      inv_num_baselines     = 1.0f/pipe->param.num_baselines;

    cudaError_t      cudaError;

    cufftHandle      cufft_plan;

    NppStreamContext npp_ctx;
    NppiSize         npp_image_size;

    size_t           i;

    fipe_pipe_cuda_callback_state pipe_state = {
        pipe,
        userdata0,
        userdata1,
        callback_input,
        callback_output,
        snap_start,
        snap_start+2,
    };


    /**
     * Select device and query its properties.
     * Plan memory allocations, but do not carry them out yet.
     * Plan CUDA kernel launch parameters.
     */

    if(fip_pipe_cuda_select_device(pipe))
        return -1;
    if(fip_pipe_cuda_plan_mem(pipe))
        return -1;
    if(fip_pipe_cuda_plan_launch(pipe))
        return -1;


    /**
     * CUDA Stream creation
     */

    if((cudaError = cudaStreamCreateWithFlags(&pipe->stream.data_read,     cudaStreamNonBlocking)) ||
       (cudaError = cudaStreamCreateWithFlags(&pipe->stream.copy_gpu,      cudaStreamNonBlocking)) ||
       (cudaError = cudaStreamCreateWithFlags(&pipe->stream.gridding,      cudaStreamNonBlocking)) ||
       (cudaError = cudaStreamCreateWithFlags(&pipe->stream.fft,           cudaStreamDefault))     || /* Internal ND bug? */
       (cudaError = cudaStreamCreateWithFlags(&pipe->stream.interpolation, cudaStreamNonBlocking)) ||
       (cudaError = cudaStreamCreateWithFlags(&pipe->stream.tlisi,         cudaStreamNonBlocking)) ||
       (cudaError = cudaStreamCreateWithFlags(&pipe->stream.copy_cpu,      cudaStreamNonBlocking)) ||
       (cudaError = cudaStreamCreateWithFlags(&pipe->stream.data_write,    cudaStreamNonBlocking))){
        fprintf(stderr, "Cannot create CUDA stream on selected device! %s (%d)\n",
                        cudaGetErrorString(cudaError),
                        (int)cudaError);
        return -1;
    }


    /**
     * CUDA Event creation
     *
     * Also initiate detailed timing here.
     */

    if(fip_pipe_cuda_create_events(pipe))
        return -1;
    if(fip_pipe_cuda_record(pipe, 0, PIPE_START, 0, 0))
        return -1;


    /**
     * NPP Context initialization
     */

    if(fip_pipe_cuda_plan_npp(pipe, &npp_image_size, &npp_ctx))
        return -1;


    /**
     * Ring Buffer and Miscellaneous memory allocations and initializations.
     */

    if(fip_pipe_cuda_alloc_mem(pipe))
        return -1;


    /**
     * cuFFT Plan creation
     */

    if(fip_pipe_cuda_plan_fft(pipe, &cufft_plan))
        return -1;


    /**
     * Main Loop
     *
     * Begin by calculating the coefficients of a convolution kernel that are static
     * for the entire duration of the loop.
     */

    fip_pipe_cuda_record(pipe, 0, LOOP_START, 0, 0);
    fip_pipe_cuda_convkernel<<<pipe->launch.Bk,
                               pipe->launch.Tk, 0,
                               pipe->stream.interpolation>>>
                                (pipe->wrkspc.conv_corr_kernel,
                                 pipe->param.image_size,
                                 pipe->param.grid_size,
                                 conv_corr_norm_factor);

    for(i=snap_start; i<snap_end; i++){
        /* Stream data_read */
        fip_pipe_cuda_record    (pipe, i, ITER_START,      pipe->stream.data_read, 0);
        fip_pipe_cuda_lock      (pipe, i, RING_VIS_PIN,    pipe->stream.data_read, 0);
        cudaLaunchHostFunc      (pipe->stream.data_read,   fip_pipe_cuda_stage_data_read, &pipe_state);
        fip_pipe_cuda_record    (pipe, i, ITER_DATA_READ,  pipe->stream.data_read, 0);


        /* Stream copy_gpu */
        fip_pipe_cuda_await     (pipe, i, ITER_DATA_READ,  pipe->stream.copy_gpu, 0);
        fip_pipe_cuda_lock      (pipe, i, RING_VIS_GPU,    pipe->stream.copy_gpu, 0);
        cudaMemcpyAsync         (fip_pipe_cuda_calc_ring_transform_gpu   (pipe, i),
                                 fip_pipe_cuda_calc_ring_transform_pinned(pipe, i),
                                 3            *            3 * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 pipe->stream.copy_gpu);
        cudaMemcpyAsync         (fip_pipe_cuda_calc_ring_vis_gpu         (pipe, i),
                                 fip_pipe_cuda_calc_ring_vis_pinned      (pipe, i),
                                 pipe->param.num_baselines   * sizeof(cufftComplex),
                                 cudaMemcpyHostToDevice,
                                 pipe->stream.copy_gpu);
        cudaMemcpyAsync         (fip_pipe_cuda_calc_ring_coords_gpu      (pipe, i),
                                 fip_pipe_cuda_calc_ring_coords_pinned   (pipe, i),
                                 pipe->param.num_baselines * 2 * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 pipe->stream.copy_gpu);
        fip_pipe_cuda_unlock    (pipe, i, RING_VIS_PIN,    pipe->stream.copy_gpu, 0);
        fip_pipe_cuda_record    (pipe, i, ITER_COPY_GPU,   pipe->stream.copy_gpu, 0);


        /* Stream gridding */
        fip_pipe_cuda_await     (pipe, i, ITER_COPY_GPU,   pipe->stream.gridding, 0);
        fip_pipe_cuda_lock      (pipe, i, RING_GRID_GPU,   pipe->stream.gridding, 0);
        cudaMemsetAsync         (fip_pipe_cuda_calc_ring_grid_gpu        (pipe, i), 0,
                                 pipe->ring.stride.grid *
                                 pipe->param.grid_size  * sizeof(cufftComplex),
                                 pipe->stream.gridding);
        fip_pipe_cuda_gridding<<<pipe->launch.Bg,
                                 pipe->launch.Tg, 0,
                                 pipe->stream.gridding>>>
                                (fip_pipe_cuda_calc_ring_grid_gpu        (pipe, i),
                                 fip_pipe_cuda_calc_ring_vis_gpu         (pipe, i), // Vis_real, Vis_imag
                                 fip_pipe_cuda_calc_ring_coords_gpu      (pipe, i), // Bin
                                 fip_pipe_cuda_calc_ring_transform_gpu   (pipe, i), // V
                                 pipe->ring.stride.grid,
                                 pipe->param.grid_size,
                                 pipe->param.num_baselines,
                                 pipe->param.cell_size * pipe->param.grid_size);
        fip_pipe_cuda_unlock    (pipe, i, RING_VIS_GPU,    pipe->stream.gridding, 0);
        fip_pipe_cuda_record    (pipe, i, ITER_GRIDDING,   pipe->stream.gridding, 0);


        /* Stream fft */
        fip_pipe_cuda_await     (pipe, i, ITER_GRIDDING,   pipe->stream.fft, 0);
        cufftExecC2C            (cufft_plan,
                                 fip_pipe_cuda_calc_ring_grid_gpu        (pipe, i),
                                 fip_pipe_cuda_calc_ring_grid_gpu        (pipe, i),
                                 CUFFT_INVERSE);
        fip_pipe_cuda_record    (pipe, i, ITER_FFT,        pipe->stream.fft, 0);


        /* Stream interpolation */
        fip_pipe_cuda_await     (pipe, i, ITER_FFT,        pipe->stream.interpolation, 0);
        fip_pipe_cuda_lock      (pipe, i, RING_IMAGE_GPU,  pipe->stream.interpolation, 0);
        cudaMemsetAsync         (fip_pipe_cuda_calc_ring_image_gpu       (pipe, i), 0,
                                 pipe->ring.stride.grid *
                                 pipe->param.grid_size  * sizeof(float),
                                 pipe->stream.interpolation);
        fip_pipe_cuda_interp<<<pipe->launch.Bs,
                               pipe->launch.Ts, 0,
                               pipe->stream.interpolation>>>
                                (fip_pipe_cuda_calc_ring_image_gpu       (pipe, i),
                                 fip_pipe_cuda_calc_ring_grid_gpu        (pipe, i),
                                 fip_pipe_cuda_calc_ring_transform_gpu   (pipe, i),
                                 pipe->ring.stride.image,
                                 pipe->param.image_size,
                                 pipe->ring.stride.grid,
                                 pipe->param.grid_size,
                                 pipe->param.cell_size,
                                 pipe->wrkspc.conv_corr_kernel,
                                 conv_corr_norm_factor,
                                 inv_num_baselines);
        fip_pipe_cuda_unlock    (pipe, i, RING_GRID_GPU,   pipe->stream.interpolation, 0);
        nppiMax_32f_C1R_Ctx     (fip_pipe_cuda_calc_ring_image_gpu       (pipe, i),
                                 pipe->ring.stride.image * sizeof(float),
                                 npp_image_size,
                                 (Npp8u*)pipe->wrkspc.npp.ptr,
                                 fip_pipe_cuda_calc_ring_max_gpu         (pipe, i),
                                 npp_ctx);
        fip_pipe_cuda_record    (pipe, i, ITER_INTERP,     pipe->stream.interpolation, 0);


        /**
         * Decision point:
         *     If fewer than 3 snapshots, we cannot yet proceed with remaining
         *     stages of the processing pipeline.
         */

        if(i < snap_start+2){
            fip_pipe_cuda_record(pipe, i, ITER_END,        pipe->stream.interpolation, 0);
            continue;
        }


        /* Stream tlisi */
        fip_pipe_cuda_await     (pipe, i, ITER_INTERP,     pipe->stream.tlisi, 0);
        fip_pipe_cuda_lock      (pipe, i, RING_RESULT_GPU, pipe->stream.tlisi, 0);
        fip_pipe_cuda_tlisi <<<pipe->launch.Bt,
                               pipe->launch.Tt,
                               pipe->launch.St,
                               pipe->stream.tlisi>>>
                                (fip_pipe_cuda_calc_ring_result_gpu      (pipe, i),
                                 fip_pipe_cuda_calc_ring_image_gpu       (pipe, i),
                                 fip_pipe_cuda_calc_ring_max_gpu         (pipe, i),
                                 fip_pipe_cuda_calc_ring_image_gpu       (pipe, i-1),
                                 fip_pipe_cuda_calc_ring_max_gpu         (pipe, i-1),
                                 fip_pipe_cuda_calc_ring_image_gpu       (pipe, i-2),
                                 fip_pipe_cuda_calc_ring_max_gpu         (pipe, i-2),
                                 pipe->ring.stride.result,
                                 pipe->ring.stride.image,
                                 pipe->param.image_size,
                                 pipe->param.unit_size,
                                 pipe->param.unit_num,
                                 C);
        fip_pipe_cuda_unlock    (pipe,i-2,RING_IMAGE_GPU,  pipe->stream.tlisi, 0);
        fip_pipe_cuda_record    (pipe, i, ITER_TLISI,      pipe->stream.tlisi, 0);


        /* Stream copy_cpu */
        fip_pipe_cuda_await     (pipe, i, ITER_TLISI,      pipe->stream.copy_cpu, 0);
        fip_pipe_cuda_lock      (pipe, i, RING_RESULT_PIN, pipe->stream.copy_cpu, 0);
        cudaMemcpy2DAsync       (fip_pipe_cuda_calc_ring_result_pinned   (pipe, i),
                                 pipe->param.unit_num     * sizeof(float),
                                 fip_pipe_cuda_calc_ring_result_gpu      (pipe, i),
                                 pipe->ring.stride.result * sizeof(float),
                                 pipe->param.unit_num     * sizeof(float),
                                 pipe->param.unit_num,
                                 cudaMemcpyDeviceToHost,
                                 pipe->stream.copy_cpu);
        fip_pipe_cuda_unlock    (pipe, i, RING_RESULT_GPU, pipe->stream.copy_cpu, 0);
        fip_pipe_cuda_record    (pipe, i, ITER_COPY_CPU,   pipe->stream.copy_cpu, 0);


        /* Stream data_write */
        fip_pipe_cuda_await     (pipe, i, ITER_COPY_CPU,   pipe->stream.data_write, 0);
        cudaLaunchHostFunc      (pipe->stream.data_write,  fip_pipe_cuda_stage_data_write, &pipe_state);
        fip_pipe_cuda_unlock    (pipe, i, RING_RESULT_PIN, pipe->stream.data_write, 0);
        fip_pipe_cuda_record    (pipe, i, ITER_DATA_WRITE, pipe->stream.data_write, 0);
        fip_pipe_cuda_record    (pipe, i, ITER_END,        pipe->stream.data_write, 0);
    }
    fip_pipe_cuda_record (pipe, 0, LOOP_END, pipe->stream.data_write, 0);

    cudaStreamSynchronize(pipe->stream.data_read);
    cudaStreamSynchronize(pipe->stream.copy_gpu);
    cudaStreamSynchronize(pipe->stream.gridding);
    cudaStreamSynchronize(pipe->stream.fft);
    cudaStreamSynchronize(pipe->stream.interpolation);
    cudaStreamSynchronize(pipe->stream.tlisi);
    cudaStreamSynchronize(pipe->stream.copy_cpu);
    cudaStreamSynchronize(pipe->stream.data_write);

    cufftDestroy(cufft_plan);

    cudaStreamDestroy(pipe->stream.data_read);
    cudaStreamDestroy(pipe->stream.copy_gpu);
    cudaStreamDestroy(pipe->stream.gridding);
    cudaStreamDestroy(pipe->stream.fft);
    cudaStreamDestroy(pipe->stream.interpolation);
    cudaStreamDestroy(pipe->stream.tlisi);
    cudaStreamDestroy(pipe->stream.copy_cpu);
    cudaStreamDestroy(pipe->stream.data_write);

    fip_pipe_cuda_free_mem(pipe);
    fip_pipe_cuda_record  (pipe, 0, PIPE_END, 0, 0);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_START],
                                        pipe->events.iter[2][ITER_DATA_READ]);
    printf("Time elapsed (Data Read):                   %10.6f ms\n", (double)milliseconds);
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_DATA_READ],
                                        pipe->events.iter[2][ITER_COPY_GPU]);
    printf("Time elapsed (Copy GPU):                    %10.6f ms\n", (double)milliseconds);
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_COPY_GPU],
                                        pipe->events.iter[2][ITER_GRIDDING]);
    printf("Time elapsed (Gridding):                    %10.6f ms\n", (double)milliseconds);
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_GRIDDING],
                                        pipe->events.iter[2][ITER_FFT]);
    printf("Time elapsed (FFT):                         %10.6f ms\n", (double)milliseconds);
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_FFT],
                                        pipe->events.iter[2][ITER_INTERP]);
    printf("Time elapsed (Interpolation):               %10.6f ms\n", (double)milliseconds);
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_INTERP],
                                        pipe->events.iter[2][ITER_TLISI]);
    printf("Time elapsed (tLISI):                       %10.6f ms\n", (double)milliseconds);
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_TLISI],
                                        pipe->events.iter[2][ITER_COPY_CPU]);
    printf("Time elapsed (Copy GPU):                    %10.6f ms\n", (double)milliseconds);
    cudaEventElapsedTime(&milliseconds, pipe->events.iter[2][ITER_COPY_CPU],
                                        pipe->events.iter[2][ITER_DATA_WRITE]);
    printf("Time elapsed (Data Write):                  %10.6f ms\n", (double)milliseconds);

    fip_pipe_cuda_destroy_events(pipe);

    return 0;
}
