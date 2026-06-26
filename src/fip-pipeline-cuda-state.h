/* Include Guard */
#ifndef SRC_FIP_PIPELINE_CUDA_STATE_H
#define SRC_FIP_PIPELINE_CUDA_STATE_H



/* Includes */
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fitsio.h>



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Typedefs */
typedef struct fip_pipeline_cuda_state fip_pipeline_cuda_state;



/* Structure Definitions */
struct fip_pipeline_cuda_state{
    /**
     * Basic Parameters
     */

    struct{
        size_t num_baselines;
        size_t image_size;
        float  cell_size;
        size_t grid_size;
        size_t unit_size;
        size_t unit_num;
    } param;


    /**
     * CUDA Device Management
     */

    struct{
        int   ordinal;   // Likely 0, may be something else on multi-GPU machines.
        struct cudaDeviceProp props; // Device properties for device with selected ordinal.
        char  name[256]; // Device name
        char  uuid[40];  // Device UUID. Format:
                         // "GPU-<8 hex>-<4 hex>-<4 hex>-<4 hex>-<12 hex>"
        char  pci [16];  // Device PCI address. Format: "DDDD:BB:DD.F", with
                         // domain not rendered if zero.
    } device;


    /**
     * CUDA Stream Management
     */

    struct{
        cudaStream_t data_read;
        cudaStream_t copy_gpu;
        cudaStream_t gridding;
        cudaStream_t fft;
        cudaStream_t interpolation;
        cudaStream_t tlisi;
        cudaStream_t copy_cpu;
        cudaStream_t data_write;
    } stream;


    /**
     * CUDA Event  Management
     */

    struct{
        cudaEvent_t  pipestart;
        cudaEvent_t  fftplan;
        cudaEvent_t  malloc;
        cudaEvent_t  coeffsready;
        cudaEvent_t  loopstart;
        cudaEvent_t  loopend;
        cudaEvent_t  pipeend;
    } evt;


    /**
     * CUDA Kernel Launch Configurations
     *
     * There are at least four typical launch configurations:
     *
     *   NAME            #THRD  #BLOCK                             SHMEM
     *   "s" (Square):   32x32, ~image_size/32 x ~image_size/32
     *   "k" (Convolve): 1024,  ~(image_size/2+1)/1024
     *   "g" (Gridding): 1024,  ~num_baselines/1024
     *   "t" (TLISI):    1024,   unit_num*unit_num                 3*1024 floats
     */

    struct{
        dim3   Ts, Bs;
        dim3   Tk, Bk;
        dim3   Tg, Bg;
        dim3   Tt, Bt;
        size_t St;
    } launch;


    /**
     * Ring Buffer Management
     *
     *   # vis_bin_pinned
     *       - Shape:    (depth=2, 2+2, num_baselines)
     *       - Depth:    Double-buffered.
     *       - Stride:   num_baselines*2*sizeof(float)
     *       - Location: CPU host memory. Pinned.
     *       - Content:
     *         - num_baselines complex single-precision values, followed by
     *         - num_baselines*2 corresponding single-precision coordinates.
     *         - Total: 4*num_baselines single-precision floats.
     *       - Designed for single cudaMemcpy2DAsync(H->D) to copy all data to GPU.
     *
     *   # vis_bin_gpu
     *       - Shape:    (depth=2, 2+2, num_baselines)
     *       - Depth:    Double-buffered.
     *       - Stride:   >= num_baselines*2*sizeof(float)
     *       - Location: GPU memory.
     *       - Content:  Identical to "vis_bin_pinned".
     *       - For simplicity, keep same depth for vis_bin_{pinned,gpu}, transform_{pinned,gpu}, and grid_gpu.
     *
     *   # transform_pinned
     *       - Shape:    (depth=2, 3, 3)
     *       - Depth:    Double-buffered.
     *       - Stride:   3*3*sizeof(float)
     *       - Location: CPU host memory. Pinned.
     *       - Content:
     *         - 3x3 transform matrix.
     *         - Total: 9 single-precision floats.
     *       - Designed for single cudaMemcpyAsync(H->D) to copy all data to GPU.
     *
     *   # transform_gpu
     *       - Shape:    (depth=2, 3, 3)
     *       - Depth:    Double-buffered.
     *       - Stride:   >= 3*3*sizeof(float), likely 64 or 128 bytes precisely.
     *       - Location: GPU memory.
     *       - Content:  Identical to "transform_pinned".
     *       - For simplicity, keep same depth for vis_bin_{pinned,gpu}, transform_{pinned,gpu}, and grid_gpu.
     *
     *   # grid_gpu
     *       - Shape:    (depth=2, grid_size, grid_size)
     *       - Depth:    Double-buffered (2) or Triple-buffered (3).
     *         - Triple-buffered requires more memory but can overlap computation
     *           with both gridding and interpolation if necessary.
     *         - [FUTURE]: Double-buffered is lighter but requires deciding if cuFFT
     *                     should be done on the gridding or interpolation streams.
     *       - Stride:   >= grid_size*2*sizeof(float)
     *       - Location: GPU memory.
     *       - Content:
     *         - grid_size x grid_size complex single-precision floats.
     *
     *   # image_gpu
     *       - Shape (depth=4, image_size, image_size)
     *       - Depth: >= 4.
     *         - The tLISI kernel requires 3 consecutive snapshots.
     *         - An in-flight interpolation kernel will be writing to a fourth.
     *       - Location: GPU memory.
     *       - Content:
     *         - grid_size x grid_size single-precision floats.
     *
     *   # max_gpu
     *       - Shape (depth=4)
     *       - Depth: >= 4.
     *         - Identical to image_gpu's.
     *       - Location: GPU memory.
     *       - Content:
     *         - One single-precision float. The maximum floating-point value of
     *           the corresponding image in the image_gpu ring buffer.
     *       - Must be kept in 1-to-1 correspondence with image_gpu.
     *
     *   # result_gpu
     *       - Shape (depth=2, unit_num, unit_num)
     *       - Depth: Double-buffered.
     *       - Location: GPU memory.
     *       - Content:
     *         - unit_num x unit_num single-precision floats.
     *
     *   # result_pinned
     *       - Shape (depth=2, unit_num, unit_num)
     *       - Depth: Double-buffered.
     *       - Location: CPU host memory. Pinned.
     *       - Identical to "result_gpu".
     *       - For simplicity, keep same depth for result_{pinned,gpu}.
     */

    struct{
        struct{
            size_t  vis_bin;   // % 2
            size_t  transform; // % 2
            size_t  grid;      // % 2 or 3
            size_t  image;     // % >= 4
            size_t  result;    // % 2
        } depth, stride;

        void*       vis_bin_pinned;
        void*       vis_bin_gpu;
        void*       transform_pinned;
        void*       transform_gpu;
        void*       grid_gpu;
        void*       image_gpu;
        void*       max_gpu;
        void*       result_gpu;
        void*       result_pinned;
    } ring;

    struct{
        struct{
            void*   ptr;
            size_t  sz;
        } cufft, npp;

        float*      conv_corr_kernel;
    } wrkspc;
};



/* Function Prototypes */
inline void fip_pipe_cuda_init(fip_pipeline_cuda_state* pipe,
                               const size_t             num_baselines,
                               const size_t             image_size,
                               const float              cell_size,
                               const size_t             unit_size,
                               const size_t             unit_num){
    memset(pipe, 0, sizeof(*pipe));
    pipe->param.num_baselines = num_baselines;
    pipe->param.grid_size     = (image_size*3+1)/2; // * 1.5, rounding up;
    pipe->param.image_size    = image_size;
    pipe->param.unit_size     = unit_size;
    pipe->param.unit_num      = unit_num;
}

int  fip_pipe_cuda     (fip_pipeline_cuda_state* pipe,
                        fitsfile* const          input,
                        fitsfile* const          output,
                        const size_t             snap_start,
                        const size_t             snap_end);



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
