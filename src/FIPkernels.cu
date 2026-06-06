#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <npp.h>

#include "utils.h"


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

#define WCSTRIG_TOL 1e-10


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

__device__ long int ceil_device(float num) {
	if (num<0) {
		return -floorf(-num);
	} else {
		return ceilf(num);
	}
}

__device__ long int floor_device(float num) {
	if (num<0) {
		return -ceilf(-num);
	} else {
		return floorf(num);
	}
}

__device__ float exp_semicircle(const float beta, float x){
	const float xx = x*x;
    
	return ((xx > float(1.0)) ? float(0.0) : exp(beta*(sqrt(float(1.0) - xx) - float(1.0))));
}

__global__ void convolveKernel(float *conv_corr_kernel, size_t image_size, size_t grid_size, float conv_corr_norm_factor) {
	const int support = 8;
	size_t t1_t2 = blockIdx.x * blockDim.x + threadIdx.x;
	if (t1_t2 < image_size / 2 + 1) {
		float t1_t2_norm = static_cast<float>(t1_t2) / grid_size;
		float correction = 0.0;
		float angle;
		for (int i = 0; i < 14; ++i) {
			angle = M_PI * t1_t2_norm * support * quadrature_nodes[i];
			correction += quadrature_kernel[i] * quadrature_weights[i] * cosf(angle);
		}
		conv_corr_kernel[t1_t2] = correction * support / conv_corr_norm_factor;
	}
}

__global__ void fused_gridding(cufftComplex* r_grid,
                               const float*  B_in,
                               const float*  Vis_real,
                               const float*  Vis_imag,
                               const float   weight,
                               const float   r1r2_scale,
                               const size_t  grid_size,
                               const size_t  num_baselines){
    const int   KERNEL_SUPPORT_BOUND = 16;
    const int   support              = 8;
    const float beta                 = 15.3704324328;
    const int   half_support         = support / 2;
    const float inv_half_support     = 1.0f / half_support;
    const long  grid_min_r1r2        = -(long)grid_size      / 2;
    const long  grid_max_r1r2        = ((long)grid_size - 1) / 2;
    const long  origin_offset_r1r2   =  (long)grid_size      / 2;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_baselines){
        float pos_r1      = B_in[idx*2+0] * r1r2_scale;
        float pos_r2      = B_in[idx*2+1] * r1r2_scale;
        long  grid_r1_min = max(ceil_device (pos_r1 - half_support), grid_min_r1r2);
        long  grid_r1_max = min(floor_device(pos_r1 + half_support), grid_max_r1r2);
        long  grid_r2_min = max(ceil_device (pos_r2 - half_support), grid_min_r1r2);
        long  grid_r2_max = min(floor_device(pos_r2 + half_support), grid_max_r1r2);
        if (grid_r1_min > grid_r1_max || grid_r2_min > grid_r2_max) {
            return;
        }
        float kernel_r1[KERNEL_SUPPORT_BOUND],
              kernel_r2[KERNEL_SUPPORT_BOUND];
        for(long grid_r1 = grid_r1_min; grid_r1 <= grid_r1_max; grid_r1++){
            kernel_r1[grid_r1 - grid_r1_min] = exp_semicircle(beta,(grid_r1 - pos_r1) * inv_half_support);
        }
        for(long grid_r2 = grid_r2_min; grid_r2 <= grid_r2_max; grid_r2++){
            kernel_r2[grid_r2 - grid_r2_min] = exp_semicircle(beta,(grid_r2 - pos_r2) * inv_half_support);
        }
        for(long grid_r1 = grid_r1_min; grid_r1 <= grid_r1_max; grid_r1++){
            for(long grid_r2 = grid_r2_min; grid_r2 <= grid_r2_max; grid_r2++){
                float kernel_value = kernel_r1[grid_r1 - grid_r1_min] * kernel_r2[grid_r2 - grid_r2_min];
                if(((grid_r1 + grid_r2) & 1) != 0){
                    kernel_value = -kernel_value;
                }
                const long grid_offset_r1r2r3 = (grid_r1 + origin_offset_r1r2) * (long)grid_size + grid_r2 + origin_offset_r1r2;
                atomicAdd(&r_grid[grid_offset_r1r2r3].x, (Vis_real[idx]/weight)*kernel_value);
                atomicAdd(&r_grid[grid_offset_r1r2r3].y, (Vis_imag[idx]/weight)*kernel_value);
            }
        }
    }
}

__global__ void fused_interpolation(float*       dirty,
                                    const cufftComplex* r_grid_stack,
                                    const float  V00, const float V01,
                                    const float  V10, const float V11,
                                    const float  V20, const float V21, const float V22,
                                    const float  dc_rad,
                                    const size_t di,
                                    const size_t gi,
                                    const float* conv_corr_kernel,
                                    const float  conv_corr_norm_factor,
                                    const float  inv_num_baselines){
    const size_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy  = blockIdx.y * blockDim.y + threadIdx.y;
    const long   half_image_size = di/2;
    const float  di2  = di*0.5f;
    const float  idxf =       idx - di2;             /* Reduced by half image size. Float */
    const float  idyf =       idy - di2;             /* Reduced by half image size. Float */
    const long   idxr = (long)idx - half_image_size; /* Reduced by half image size. Integer */
    const long   idyr = (long)idy - half_image_size; /* Reduced by half image size. Integer */
    const float  r0   = 180.0f / M_PI;
    const float  dc   = dc_rad / M_PI * 180;
    const float  xi   = V20/V22;
    const float  eta  = V21/V22;


    float        oi0, oi1;                           /* (ex-) output_index[k+0], output_index[k+1] */
    float        pixel_sum;                          /* (ex-) dirty_pre[idy*di + idx] */

    dirty   +=   half_image_size*di + half_image_size;

    if(idx<di && idy<di){
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

        pixel_sum = r_grid_stack[gi*gi/2 + gi/2 + idyr*(long)gi + idxr].x;
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

        const float LL    = oi0 - half_image_size;
        const float MM    = oi1 - half_image_size;
        const float value = pixel_sum * inv_num_baselines;

        if(fabs(LL) < half_image_size-1 && fabs(MM)<half_image_size-1){
            const float LLf = floorf(LL);
            const float MMf = floorf(MM);
            const float LLc = ceilf (LL);/* Theoretically LLf+1 except if LL was integer */
            const float MMc = ceilf (MM);/* Theoretically MMf+1 except if MM was integer */

            atomicAdd(&dirty[(long)MMf * (long)di + (long)LLf],  (1-LL+LLf) * (1-MM+MMf) * value);/* Always effective                  */
            atomicAdd(&dirty[(long)MMc * (long)di + (long)LLf],  (1-LL+LLf) * (0+MM-MMf) * value);/* Ineffective when       MM integer */
            atomicAdd(&dirty[(long)MMf * (long)di + (long)LLc],  (0+LL-LLf) * (1-MM+MMf) * value);/* Ineffective when LL       integer */
            atomicAdd(&dirty[(long)MMc * (long)di + (long)LLc],  (0+LL-LLf) * (0+MM-MMf) * value);/* Ineffective when LL or MM integer */
        }
    }
}

__global__ void tlisi2(float* result,
                       const float* diff_out,
                       const float* snap,
                       const size_t unit_size,
                       const size_t ima,
                       const size_t unit_num,
                       const float* maxall){
    extern  __shared__  float sharedNumDen[];

    const float  maxallval = max(maxall[0], max(maxall[1], maxall[2]));
    const size_t bid       = blockIdx.x; // tile index
    const size_t tid       = threadIdx.x;

    const size_t i_id      = bid / unit_num;
    const size_t j_id      = bid % unit_num;
    const size_t factor    = (size_t)ceil_device((float)(unit_size * unit_size)/1024.0f);

    for(size_t f=0; f<factor; f++){
        if(tid+f*1024 < unit_size*unit_size){
            if(f == 0){
                sharedNumDen[tid+   0] = 0; /* Sum of diff_out */
                sharedNumDen[tid+1024] = 0; /* Max of diff_out */
                sharedNumDen[tid+2048] = 0; /* Sum of r        */
            }
            size_t rows = (tid + f*1024) / unit_size;
            size_t cols = (tid + f*1024) % unit_size;

            size_t I_id = i_id * unit_size + rows;
            size_t J_id = j_id * unit_size + cols;

            sharedNumDen[tid+   0] =     sharedNumDen[tid+   0] + diff_out[I_id * ima + J_id];
            sharedNumDen[tid+1024] = max(sharedNumDen[tid+1024],  diff_out[I_id * ima + J_id]);
            sharedNumDen[tid+2048] =                             (diff_out[I_id * ima + J_id] / snap[I_id * ima + J_id] < 1) ?
                                         sharedNumDen[tid+2048] + diff_out[I_id * ima + J_id] / snap[I_id * ima + J_id] :
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
        result[bid] = 1 - (sharedNumDen[0   ]/unit_size/unit_size) *
                           sharedNumDen[1024]                      *
                          (sharedNumDen[2048]/unit_size/unit_size) / maxallval / maxallval;
    }
}



/*************************************************************************/

/**
 * @brief Ceiling Divide.
 *
 * Perform a/b, rounding up.
 *
 * @param [in]  a  Dividend.
 * @param [in]  b  Divisor. Undefined behaviour if 0.
 * @return Quotient, rounded up to nearest integer.
 */

size_t ceiling_divide(size_t a, size_t b) {
    size_t q =  a/b;
    return q + (a > q*b);
}

/* New FI pipe. */
int FIpipe2(float* Visreal,
            float* Visimag,
            float* Bin,
            float* Vin,
            float* result_array,
            size_t num_baselines,
            size_t image_size,
            size_t num_snapshots,
            float  cell_size,
            size_t unit_size){
    float *Vis_real, *Vis_imag, *B_in;
    float *Vis_realtmp, *Vis_imagtmp, *B_intmp;
    float *pinned_Vis_real, *pinned_Vis_imag, *pinned_B_in;
    float *dirty1, *dirty2, *dirty3, *dirtyp;
    float *conv_corr_kernel, *maxall;
    float* image_buffer;
    float* Vis_buffer;
    float (*V)[3];

    float* d_data_1;
    float* d_data_2;
    float* diff_out;
    float* result_data;

    size_t i, ind;
    float  milliseconds=0, milliseconds1=0;

    const size_t num_events  = num_snapshots>3 ? num_snapshots : 3;
    const size_t unit_num    = image_size/unit_size;
    const size_t grid_size   = ceiling_divide(image_size*3, 2); // * 1.5, rounding up

    const float r1r2_scale   = cell_size*grid_size;
    const float conv_corr_norm_factor = 2.4937047051153827;
    const float C            = 1e-6;

    cudaError_t    cudaError;
    cufftResult    cufftError;
    int            cudaOrdinal;
    cudaDeviceProp cudaDevProps;
    cufftComplex*  r_grid_stack;
    cudaStream_t   stream1, stream2, stream3;
    cufftHandle    plan;
    cudaEvent_t    start, stop, *eventstream, *events, *events_kernel;
    cudaEvent_t    start1, stop1;

    NppiSize         nppImageSize = {(int)image_size, (int)image_size};
    NppStreamContext nppCtx1, nppCtx2, nppCtxt;
    size_t           nppWrkspc1Sz;
    Npp8u*           nppWrkspc1;


    /* Device Selection and Property Query */
    if((cudaError = cudaGetDevice(&cudaOrdinal))){
        printf("Cannot find CUDA device (%d)\n", (int)cudaError);
        return -1;
    }
    if((cudaError = cudaGetDeviceProperties(&cudaDevProps, cudaOrdinal))){
        printf("Cannot get the properties of CUDA device with ordinal %d (%d)\n",
               (int)cudaOrdinal,
               (int)cudaError);
        return -1;
    }else{
        printf("Selected GPU %d: %s (UUID: "
               "GPU-%02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-"
                   "%02hhx%02hhx-%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx, "
               "PCIe [%04x:]%02x:%02x.0)\n",
               cudaOrdinal,
               cudaDevProps.name,
               cudaDevProps.uuid.bytes[ 0], cudaDevProps.uuid.bytes[ 1],
               cudaDevProps.uuid.bytes[ 2], cudaDevProps.uuid.bytes[ 3],
               cudaDevProps.uuid.bytes[ 4], cudaDevProps.uuid.bytes[ 5],
               cudaDevProps.uuid.bytes[ 6], cudaDevProps.uuid.bytes[ 7],
               cudaDevProps.uuid.bytes[ 8], cudaDevProps.uuid.bytes[ 9],
               cudaDevProps.uuid.bytes[10], cudaDevProps.uuid.bytes[11],
               cudaDevProps.uuid.bytes[12], cudaDevProps.uuid.bytes[13],
               cudaDevProps.uuid.bytes[14], cudaDevProps.uuid.bytes[15],
               cudaDevProps.pciDomainID,
               cudaDevProps.pciBusID,
               cudaDevProps.pciDeviceID);
    }
    if(cudaDevProps.maxThreadsPerBlock < 1024){
        printf("Selected CUDA device supports fewer than 1024 threads/block! (%d)\n",
               cudaDevProps.maxThreadsPerBlock);
        return -1;
    }


    /* CUDA Stream creations */
    if((cudaError = cudaStreamCreate(&stream1)) != cudaSuccess ||
       (cudaError = cudaStreamCreate(&stream2)) != cudaSuccess ||
       (cudaError = cudaStreamCreate(&stream3)) != cudaSuccess){
        printf("Cannot create CUDA stream on selected device! (%d)\n", (int)cudaError);
        return -1;
    }


    /* cuFFT Plan creation */
    if((cufftError = cufftCreate(&plan))){
        printf("Cannot create cuFFT plan! (%d)\n", (int)cufftError);
        return -1;
    }
    if((cufftError = cufftSetStream(plan, stream1))){
        printf("Cannot assign stream to cuFFT plan! (%d)\n", (int)cufftError);
        return -1;
    }
    if((cufftError = cufftPlan2d(&plan, grid_size, grid_size, CUFFT_C2C))){
        printf("Cannot make cuFFT plan for grid of size %zu (%d)\n", grid_size, (int)cufftError);
        return -1;
    }


    /* NPP Context initializations */
    nppGetStreamContext(&nppCtxt);
    nppCtx2 = nppCtx1 = nppCtxt;
    nppCtx1.hStream                            = stream1;
    cudaStreamGetFlags(nppCtx1.hStream, &nppCtx1.nStreamFlags);
    nppCtx2.hStream                            = stream2;
    cudaStreamGetFlags(nppCtx2.hStream, &nppCtx2.nStreamFlags);
    nppiMaxGetBufferHostSize_32f_C1R_Ctx(nppImageSize, &nppWrkspc1Sz, nppCtx1);


    /* Consolidated memory allocations and initializations. */
    cudaMalloc((void**)&nppWrkspc1,          nppWrkspc1Sz);

    cudaMalloc((void**)&image_buffer,        4 * image_size * image_size * sizeof(float));
    cudaMalloc((void**)&r_grid_stack,            grid_size  * grid_size  * sizeof(cufftComplex));
    cudaMalloc((void**)&Vis_buffer,          8 * num_baselines           * sizeof(float));

    cudaMalloc((void**)&conv_corr_kernel,   (image_size/2+1)             * sizeof(float));
    cudaMalloc((void**)&maxall,              3                           * sizeof(float));

    cudaMallocHost((void**)&pinned_Vis_real, num_baselines *     num_snapshots * sizeof(float));
    cudaMallocHost((void**)&pinned_Vis_imag, num_baselines *     num_snapshots * sizeof(float));
    cudaMallocHost((void**)&pinned_B_in,     num_baselines * 2 * num_snapshots * sizeof(float));
    memcpy(pinned_Vis_real, Visreal,         num_baselines *     num_snapshots * sizeof(float));
    memcpy(pinned_Vis_imag, Visimag,         num_baselines *     num_snapshots * sizeof(float));
    memcpy(pinned_B_in,     Bin,             num_baselines * 2 * num_snapshots * sizeof(float));

    cudaMemset(image_buffer,  0,  4*image_size*image_size * sizeof(float));
    dirty1       = image_buffer + 1*image_size*image_size;
    dirty2       = image_buffer + 2*image_size*image_size;
    dirty3       = image_buffer + 3*image_size*image_size;

    Vis_realtmp  = Vis_buffer + 0*num_baselines;
    Vis_imagtmp  = Vis_buffer + 1*num_baselines;
    B_intmp      = Vis_buffer + 2*num_baselines;
    Vis_real     = Vis_buffer + 4*num_baselines;
    Vis_imag     = Vis_buffer + 5*num_baselines;
    B_in         = Vis_buffer + 6*num_baselines;

    cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess){
        printf("ERROR! GPU Kernel 1 error.\n");
        printf("CUDA error code: %d; string: %s;\n", (int)cudaError, cudaGetErrorString(cudaError));
    }else{
        printf("No CUDA error 1.\n");
    }


    /**
     * There are at least six typical launch configurations:
     *
     *   NAME            #THRD  #BLOCK                             SHMEM
     *   "s" (Square):   32x32, ~image_size/32 x ~image_size/32
     *   "k" (Convolve): 1024,  ~(image_size/2+1)/1024
     *   "g" (Gridding): 1024,  ~num_baselines/1024
     *   "t" (TLISI):    1024,   unit_num*unit_num                 3*1024 floats
     *
     * Abbreviate them and centralize their calculations here.
     */

    dim3 Ts = {32, 32}, Bs = {
        (unsigned)ceiling_divide(image_size, Ts.x),
        (unsigned)ceiling_divide(image_size, Ts.y),
    };
    dim3 Tk = {1024},   Bk = {
        (unsigned)ceiling_divide(image_size/2+1, Tk.x)
    };
    dim3 Tg = {1024},   Bg = {
        (unsigned)ceiling_divide(num_baselines,  Tg.x)
    };
    dim3 Tt = {1024},   Bt = {(unsigned)(unit_num*unit_num)};

    size_t St = 3 * Tt.x * sizeof(float);


    /* CUDA Event creation */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    eventstream   = (cudaEvent_t*)calloc(num_events, sizeof(cudaEvent_t));
    events        = (cudaEvent_t*)calloc(num_events, sizeof(cudaEvent_t));
    events_kernel = (cudaEvent_t*)calloc(num_events, sizeof(cudaEvent_t));
    for(i=0; i<num_events; i++){
        cudaEventCreate(&eventstream[i]);
        cudaEventCreate(&events[i]);
        cudaEventCreate(&events_kernel[i]);
    }


    /**
     * Main Loop
     *
     * Begin by calculating the coefficients of a convolution kernel that are static
     * for the entire duration of the loop. Exclude this from timing.
     */

    convolveKernel <<<Bk, Tk>>> (conv_corr_kernel, image_size, grid_size, conv_corr_norm_factor);
    cudaEventRecord(start);
    /* ****************************************************** */
    for(ind=0, V=((float(*)[3])Vin)-3; ind<num_snapshots; ind++, V+=3){
        cudaMemcpyAsync(Vis_realtmp, pinned_Vis_real + ind*num_baselines,   num_baselines*1*sizeof(float), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(Vis_imagtmp, pinned_Vis_imag + ind*num_baselines,   num_baselines*1*sizeof(float), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(B_intmp,     pinned_B_in     + ind*num_baselines*2, num_baselines*2*sizeof(float), cudaMemcpyHostToDevice, stream3);

        if(ind == 0){
            cudaMemcpyAsync(Vis_real, Vis_realtmp, num_baselines*1*sizeof(float), cudaMemcpyDeviceToDevice, stream3);
            cudaMemcpyAsync(Vis_imag, Vis_imagtmp, num_baselines*1*sizeof(float), cudaMemcpyDeviceToDevice, stream3);
            cudaMemcpyAsync(B_in,     B_intmp,     num_baselines*2*sizeof(float), cudaMemcpyDeviceToDevice, stream3);
            cudaEventRecord(events[ind],                                                                    stream3);
        }else{
            cudaStreamWaitEvent(stream1, events[ind-1], 0);
            cudaStreamWaitEvent(stream2, events[ind-1], 0);

            cudaMemsetAsync(r_grid_stack, 0, grid_size  * grid_size   *sizeof(cufftComplex), stream1);

            fused_gridding <<<Bg, Tg, 0, stream1>>> (r_grid_stack, B_in, Vis_real, Vis_imag,
                                                     fabsf(V[0][0]*V[1][1] - V[0][1]*V[1][0]),
                                                     r1r2_scale, grid_size, num_baselines);

            cufftExecC2C(plan, r_grid_stack, r_grid_stack, CUFFT_INVERSE);

            cudaEventRecord    (eventstream[ind-1], stream2);
            cudaStreamWaitEvent(stream1, eventstream[ind-1], 0);

            if(ind == 1 || ind == 2){
                dirtyp = ind == 1 ? dirty1 : dirty2;
                fused_interpolation <<<Bs, Ts, 0,  stream1>>> (dirtyp,
                                                               r_grid_stack,
                                                               V[0][0], V[0][1],
                                                               V[1][0], V[1][1],
                                                               V[2][0], V[2][1], V[2][2],
                                                               cell_size, image_size, grid_size,
                                                               conv_corr_kernel,
                                                               conv_corr_norm_factor,
                                                               1.0f/num_baselines);
                nppiMax_32f_C1R_Ctx(dirtyp, image_size*sizeof(float), nppImageSize, nppWrkspc1, maxall+(ind-1), nppCtx1);
            }

            cudaEventRecord(events_kernel[ind],stream1);
            cudaStreamWaitEvent(stream3,events_kernel[ind],0);
            cudaMemcpyAsync(Vis_real, Vis_realtmp, num_baselines*1*sizeof(float), cudaMemcpyDeviceToDevice, stream3);
            cudaMemcpyAsync(Vis_imag, Vis_imagtmp, num_baselines*1*sizeof(float), cudaMemcpyDeviceToDevice, stream3);
            cudaMemcpyAsync(B_in,     B_intmp,     num_baselines*2*sizeof(float), cudaMemcpyDeviceToDevice, stream3);
            cudaEventRecord(events[ind], stream3);
        }
    }

    cudaStreamWaitEvent(stream1, events[ind-1], 0);
    cudaStreamWaitEvent(stream2, events[ind-1], 0);

    cudaMemsetAsync(r_grid_stack, 0, grid_size  * grid_size   *sizeof(cufftComplex), stream1);

    fused_gridding <<<Bg, Tg, 0, stream1>>> (r_grid_stack, B_in, Vis_real, Vis_imag,
                                             fabsf(V[0][0]*V[1][1] - V[0][1]*V[1][0]),
                                             r1r2_scale, grid_size, num_baselines);

    cufftExecC2C(plan, r_grid_stack, r_grid_stack, CUFFT_INVERSE);

    cudaEventRecord    (eventstream[ind-1], stream2);
    cudaStreamWaitEvent(stream1, eventstream[ind-1], 0);
    fused_interpolation <<<Bs, Ts, 0,  stream1>>> (dirty3,
                                                   r_grid_stack,
                                                   V[0][0], V[0][1],
                                                   V[1][0], V[1][1],
                                                   V[2][0], V[2][1], V[2][2],
                                                   cell_size, image_size, grid_size,
                                                   conv_corr_kernel,
                                                   conv_corr_norm_factor,
                                                   1.0f/num_baselines);
    nppiMax_32f_C1R_Ctx(dirty3, image_size*sizeof(float), nppImageSize, nppWrkspc1, maxall+(ind-1), nppCtx1);

    cudaStreamSynchronize(stream1);

    /* ****************************************************** */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %9.6f ms\n", (double)milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for(i=0; i<num_events; i++){
        cudaEventDestroy(eventstream[i]);
        cudaEventDestroy(events[i]);
        cudaEventDestroy(events_kernel[i]);
    }
    free(eventstream);
    free(events);
    free(events_kernel);
    eventstream   = NULL;
    events        = NULL;
    events_kernel = NULL;

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    cudaFree(r_grid_stack);
    cudaFree(Vis_buffer);
    cudaFree(conv_corr_kernel);

    cudaFreeHost(pinned_Vis_real);
    cudaFreeHost(pinned_Vis_imag);
    cudaFreeHost(pinned_B_in);


    // FI Trigger
    cudaMalloc((void**)&d_data_1,    image_size*image_size*sizeof(float));
    cudaMalloc((void**)&d_data_2,    image_size*image_size*sizeof(float));
    cudaMalloc((void**)&diff_out,    image_size*image_size*sizeof(float));
    cudaMalloc((void**)&result_data, unit_num  *unit_num  *sizeof(float));

    cudaMemset(d_data_1,    0,       image_size*image_size*sizeof(float));
    cudaMemset(d_data_2,    0,       image_size*image_size*sizeof(float));
    cudaMemset(diff_out,    0,       image_size*image_size*sizeof(float));
    cudaMemset(result_data, 0,       unit_num  *unit_num  *sizeof(float));

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    /* ****************************************************** */
    nppiAbsDiff_32f_C1R_Ctx        (dirty1,   image_size*sizeof(float),
                                    dirty2,   image_size*sizeof(float),
                                    d_data_1, image_size*sizeof(float),
                                    nppImageSize,          nppCtxt);
    nppiAbsDiff_32f_C1R_Ctx        (dirty2,   image_size*sizeof(float),
                                    dirty3,   image_size*sizeof(float),
                                    d_data_2, image_size*sizeof(float),
                                    nppImageSize,          nppCtxt);
    nppiThreshold_LTVal_32f_C1R_Ctx(dirty2,   image_size*sizeof(float),
                                    dirty2,   image_size*sizeof(float), // (In-place)
                                    nppImageSize,
                                    nextafterf(0.0f, 1.0f),           // LTVal uses < and we want <=
                                    C,
                                    nppCtxt);
    nppiAbsDiff_32f_C1R_Ctx        (d_data_1, image_size*sizeof(float),
                                    d_data_2, image_size*sizeof(float),
                                    diff_out, image_size*sizeof(float),
                                    nppImageSize,          nppCtxt);
    tlisi2 <<<Bt, Tt, St>>> (result_data, diff_out, dirty2, unit_size, image_size, unit_num, maxall);

    /* ****************************************************** */
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    printf("Time elapsed: %9.6f ms\n", (double)milliseconds1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    cudaMemcpy(result_array, result_data, unit_num * unit_num * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(image_buffer);
    cudaFree(maxall);

    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaFree(diff_out);
    cudaFree(result_data);

    cudaFree(nppWrkspc1);

    return 0;
}
