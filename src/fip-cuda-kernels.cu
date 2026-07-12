#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "fip-cuda-kernels.h"


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

__global__ void convkernel     (float*           conv_corr_kernel,
                                size_t           image_size,
                                size_t           grid_size,
                                float            conv_corr_norm_factor){
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

__global__ void gridding       (cuComplex*       grid,
                                const cuComplex* visibilities,
                                const float*     coords,
                                const float      transform[3][3],
                                const size_t     grid_stride,
                                const size_t     grid_size,
                                const size_t     num_baselines,
                                const float      r1r2_scale){
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
    cuComplex*    grid_origin          = &grid[grid_stride*grid_size_half +
                                                              grid_size_half];
    cuComplex     v;
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

__global__ void interp         (float*           image,
                                const cuComplex* grid,
                                const float      transform[3][3],
                                const size_t     image_stride,
                                const size_t     image_size,
                                const size_t     grid_stride,
                                const size_t     grid_size,
                                const float      dc_rad,
                                const float*     conv_corr_kernel,
                                const float      conv_corr_norm_factor,
                                const float      inv_num_baselines){
    const long          image_stride_l    =  image_stride;
    const size_t        image_size_half   =  image_size/2;
    const long          image_size_half_l =  image_size_half;
    float*              image_origin      = &image[image_size_half*image_stride + image_size_half];
    const long          grid_stride_l     =  grid_stride;
    const size_t        grid_size_half    =  grid_size/2;
    const cuComplex*    grid_origin       = &grid[grid_size_half  *grid_stride  + grid_size_half];

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

__global__ static void tlisi   (float*           result,
                                const float*     image0,
                                const float*     max0,
                                const float*     image1,
                                const float*     max1,
                                const float*     image2,
                                const float*     max2,
                                const size_t     result_stride,
                                const size_t     image_stride,
                                const size_t     image_size,
                                const size_t     unit_size,
                                const size_t     unit_num,
                                const float      C){
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

/**
 * Shims to invoke CUDA kernels easily from C.
 */


void fip_cuda_kernel_convkernel    (dim3                cuda_grid,
                                    dim3                cuda_thrd,
                                    size_t              cuda_shmem,
                                    cudaStream_t        cuda_stream,
                                    float*              conv_corr_kernel,
                                    size_t              image_size,
                                    size_t              grid_size,
                                    float               conv_corr_norm_factor){
    convkernel<<<cuda_grid,
                 cuda_thrd,
                 cuda_shmem,
                 cuda_stream>>>(
        conv_corr_kernel,
        image_size,
        grid_size,
        conv_corr_norm_factor
    );
}

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
                                    const float         r1r2_scale){
    gridding<<<cuda_grid,
               cuda_thrd,
               cuda_shmem,
               cuda_stream>>>(
        (cuComplex*)grid,
        (const cuComplex*)visibilities,
        coords,
        transform,
        grid_stride,
        grid_size,
        num_baselines,
        r1r2_scale
    );
}

void fip_cuda_kernel_interp        (dim3                cuda_grid,
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
                                    const float         inv_num_baselines){
    interp<<<cuda_grid,
             cuda_thrd,
             cuda_shmem,
             cuda_stream>>>(
        image,
        (const cuComplex*)grid,
        transform,
        image_stride,
        image_size,
        grid_stride,
        grid_size,
        dc_rad,
        conv_corr_kernel,
        conv_corr_norm_factor,
        inv_num_baselines
    );
}

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
                                    const float         C){
    tlisi<<<cuda_grid,
            cuda_thrd,
            cuda_shmem,
            cuda_stream>>>(
        result,
        image0,
        max0,
        image1,
        max1,
        image2,
        max2,
        result_stride,
        image_stride,
        image_size,
        unit_size,
        unit_num,
        C
    );
}
