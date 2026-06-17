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
typedef struct fip_pipe_cuda_state fip_pipe_cuda_state;


/* Enums */
enum fip_pipe_cuda_ring{
    RING_VIS_PIN,
    RING_VIS_GPU,
    RING_GRID_GPU,
    RING_IMAGE_GPU,
    RING_RESULT_GPU,
    RING_RESULT_PIN,
};
typedef enum fip_pipe_cuda_ring fip_pipe_cuda_ring;

enum fip_pipe_cuda_event{
    /* Pre-loop Events */
    PIPE_START         = -2,
    LOOP_START,

    /* Loop Events */
    ITER_START         =  0,   /* Loop iteration (i) start */
    ITER_DATA_READ,            /* Data has been read into pinned input buffer */
    ITER_COPY_GPU,             /* Data has been copied to GPU */
    ITER_GRIDDING,             /* Gridding      complete */
    ITER_FFT,                  /* FFT           complete */
    ITER_INTERP,               /* Interpolation complete */
    ITER_TLISI,                /* tLISI         executed */
    ITER_COPY_CPU,             /* Data has been copied from GPU to pinned output buffer */
    ITER_DATA_WRITE,           /* Data has been written out of pinned output buffer */
    ITER_END,                  /* Loop iteration (i) end.
                                  Not identical to ITER_DATA_WRITE for first 2 iterations. */
    ITER_NUM_EVENTS,           /* Number of loop events being recorded. */

    /* Post-loop Events */
    LOOP_END           = ITER_NUM_EVENTS,
    PIPE_END,
};
typedef enum fip_pipe_cuda_event fip_pipe_cuda_event;



/* Structure Definitions */
struct fip_pipe_cuda_state{
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
        char  uuid[48];  // Device UUID. Format:
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
        cudaEvent_t  iter[4][ITER_NUM_EVENTS];
        cudaEvent_t  loopend;
        cudaEvent_t  pipeend;
        struct{
            cudaEvent_t* vis_pinned;
            cudaEvent_t* vis_gpu;
            cudaEvent_t* grid_gpu;
            cudaEvent_t* image_gpu;
            cudaEvent_t* result_gpu;
            cudaEvent_t* result_pinned;
        } ring;
    } events;


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
     *       - Shape:    (depth=2, 2+2|3, num_baselines)
     *       - Depth:    Double-buffered.
     *       - Dtype:    cuComplex + float[2|3]
     *       - Stride:   num_baselines
     *       - Location: CPU host memory. Pinned.
     *       - Content:
     *         - num_baselines complex single-precision values, followed by
     *         - num_baselines*2 (or *3) corresponding single-precision coordinates.
     *         - Total: 4 or 5 * num_baselines single-precision floats.
     *
     *   # vis_bin_gpu
     *       - Shape:    (depth=2, 2+2, num_baselines)
     *       - Depth:    Double-buffered.
     *       - Dtype:    cuComplex + float[2|3]
     *       - Stride:   >= num_baselines
     *       - Location: GPU memory.
     *       - Content:  Identical to "vis_bin_pinned".
     *       - For simplicity, keep same depth for vis_bin_{pinned,gpu}, transform_{pinned,gpu}, and grid_gpu.
     *
     *   # transform_pinned
     *       - Shape:    (depth=2, 3, 3)
     *       - Depth:    Double-buffered.
     *       - Dtype:    float
     *       - Stride:   3*3
     *       - Location: CPU host memory. Pinned.
     *       - Content:
     *         - 3x3 transform matrix.
     *         - Total: 9 single-precision floats.
     *       - Designed for single cudaMemcpyAsync(H->D) to copy all data to GPU.
     *
     *   # transform_gpu
     *       - Shape:    (depth=2, 3, 3)
     *       - Depth:    Double-buffered.
     *       - Dtype:    float
     *       - Stride:   >= 3*3, likely 16 or 32 elements (64 or 128 bytes) precisely.
     *       - Location: GPU memory.
     *       - Content:  Identical to "transform_pinned".
     *       - For simplicity, keep same depth for vis_bin_{pinned,gpu} and transform_{pinned,gpu}.
     *
     *   # grid_gpu
     *       - Shape:    (depth=2, grid_size, grid_size)
     *       - Depth:    Double-buffered (2) or Triple-buffered (3).
     *         - Triple-buffered requires more memory but can overlap computation
     *           with both gridding and interpolation if necessary.
     *         - [FUTURE]: Double-buffered is lighter but requires deciding if cuFFT
     *                     should be done on the gridding or interpolation streams.
     *       - Dtype:    cuComplex
     *       - Stride:   >= grid_size
     *       - Location: GPU memory.
     *       - Content:
     *         - grid_size x grid_size complex single-precision floats.
     *
     *   # image_gpu
     *       - Shape (depth=4, image_size, image_size)
     *       - Depth: >= 4.
     *         - The tLISI kernel requires 3 consecutive snapshots.
     *         - An in-flight interpolation kernel will be writing to a fourth.
     *       - Dtype:    float
     *       - Stride:   >= image_size
     *       - Location: GPU memory.
     *       - Content:
     *         - image_size x image_size single-precision floats.
     *
     *   # max_gpu
     *       - Shape (depth=4)
     *       - Depth: >= 4.
     *         - Identical to image_gpu's.
     *       - Dtype:    float
     *       - Stride:   1
     *       - Location: GPU memory.
     *       - Content:
     *         - One single-precision float. The maximum floating-point value of
     *           the corresponding image in the image_gpu ring buffer.
     *       - Must be kept in 1-to-1 correspondence with image_gpu.
     *
     *   # result_gpu
     *       - Shape (depth=2, unit_num, unit_num)
     *       - Depth: Double-buffered.
     *       - Dtype:    float
     *       - Stride:   >= unit_num
     *       - Location: GPU memory.
     *       - Content:
     *         - unit_num x unit_num single-precision floats.
     *
     *   # result_pinned
     *       - Shape (depth=2, unit_num, unit_num)
     *       - Depth: Double-buffered.
     *       - Dtype:    float
     *       - Stride:   unit_num
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
#if defined(__cplusplus)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
inline void fip_pipe_cuda_init(fip_pipe_cuda_state* pipe,
                               const size_t             num_baselines,
                               const size_t             image_size,
                               const float              cell_size,
                               const size_t             unit_size,
                               const size_t             unit_num){
    memset(pipe, 0, sizeof(*pipe));
    pipe->param.num_baselines = num_baselines;
    pipe->param.grid_size     = (image_size*3+1)/2; // * 1.5, rounding up;
    pipe->param.image_size    = image_size;
    pipe->param.cell_size     = cell_size;
    pipe->param.unit_size     = unit_size;
    pipe->param.unit_num      = unit_num;
}
#if defined(__cplusplus)
#pragma GCC diagnostic pop
#endif

typedef void (*fip_pipe_cuda_input_cb) (void*  userdata0,
                                        void*  userdata1,
                                        void*  visibilities,
                                        float* coords,
                                        float  transform[3][3],
                                        size_t num_baselines,
                                        size_t iter);
typedef void (*fip_pipe_cuda_output_cb)(void*  userdata0,
                                        void*  userdata1,
                                        void*  result,
                                        size_t unit_num,
                                        size_t iter);

int  fip_pipe_cuda     (fip_pipe_cuda_state*     pipe,
                        fip_pipe_cuda_input_cb   callback_input,
                        fip_pipe_cuda_output_cb  callback_output,
                        void*                    userdata0,
                        void*                    userdata1,
                        const size_t             snap_start,
                        const size_t             snap_end);



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
