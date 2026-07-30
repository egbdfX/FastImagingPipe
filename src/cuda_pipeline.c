/* Includes */
#include <stdio.h>


/**
 * CUDA defines enumerator values beyond the range of int, which is invalid
 * under ISO C older than C23.
 *
 * Because these headers cannot be changed, reduce the noise by silencing the
 * warning when feasible under GCC 4.6+, or Clang 3+. This can be done by
 * creating a temporary context where the -Wpedantic diagnostic is ignored,
 * #include'ing cuda.h, then destroying this context.
 */

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) || \
    __clang_major__ >= 3
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <cuda.h>

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) || \
    __clang_major__ >= 3
#pragma GCC diagnostic pop
#endif

#include <cuda_runtime.h>
#include <device_types.h>
#include <cufft.h>
#include <npp.h>

#include "fip/cuda_kernels.h"
#include "fip/cuda_pipeline.h"


/* Defines */

/**
 * cudaEventRecordWithFlags() was only introduced in CUDA 11.2+.
 * Silently fall back to cudaEventRecord() in its absence.
 * Also silence the "unused" warning about flags in that case.
 */

#if CUDART_VERSION < 11020
#define cudaEventRecordWithFlags(event, stream, flags)  \
        ((void)(flags), cudaEventRecord((event), (stream)))
#endif



/* Enums */
enum fip_pipe_cuda_ring{
    RING_VIS_PIN,
    RING_VIS_GPU,
    RING_XFORM_GPU,
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
        int    verbose;
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
            cudaEvent_t* transform_gpu;
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
     *       - For simplicity, keep same depth for vis_bin_{pinned,gpu} and transform_pinned.
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
     *       - Shape:    (depth=8, 3, 3)
     *       - Depth:    Octuple-buffered.
     *       - Dtype:    float
     *       - Stride:   16 (next NAPOT >= 3*3), 64 bytes precisely.
     *       - Location: GPU memory.
     *       - Content:  Identical to "transform_pinned".
     *       - Because transform_gpu is required for more than just the first stage
     *         of processing, gridding (it is also required at the interpolation stage),
     *         transform_gpu must be retained longer than either vis_bin_gpu or
     *         transform_pinned, and deserves its own ring. Because it is anyways
     *         fairly small, even a deep 8-entry ring, dedicating 64 bytes per 3x3
     *         transform matrix, is not particularly onerous (512 bytes).
     *       - Depth must be at least +2 greater than vis_bin.
     *
     *   # grid_gpu
     *       - Shape:    (depth=3, grid_size, grid_size)
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
            size_t  vis_bin;       // % 2
            size_t  transform_gpu; // % 8
            size_t  grid;          // % 2 or 3
            size_t  image;         // % >= 4
            size_t  result;        // % 2
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

        void*       conv_corr_kernel;
    } wrkspc;
};




/* Utility functions */

/**
 * @brief Ceiling Divide.
 *
 * Perform a/b, rounding up.
 *
 * @param [in]  a  Dividend.
 * @param [in]  b  Divisor. Undefined behaviour if 0.
 * @return Quotient, rounded up to nearest integer.
 */

static size_t        ceiling_divide(size_t a, size_t b) {
    size_t q =  a/b;
    return q + (a > q*b);
}



/* FIP CUDA pipeline functions */

int                  fip_pipe_cuda_alloc                     (fip_pipe_cuda_state**    pipe_ptr,
                                                              const int                verbose,
                                                              const int                gpu_ordinal,
                                                              const size_t             num_baselines,
                                                              const size_t             image_size,
                                                              const float              cell_size,
                                                              const size_t             unit_size,
                                                              const size_t             unit_num){
    fip_pipe_cuda_state* pipe;

    if(!pipe_ptr || !(*pipe_ptr = pipe = calloc(1, sizeof(*pipe))))
        return -1;

    pipe->param.verbose       = verbose;
    pipe->device.ordinal      = gpu_ordinal;
    pipe->param.num_baselines = num_baselines;
    pipe->param.grid_size     = (image_size*3+1)/2; // * 1.5, rounding up;
    pipe->param.image_size    = image_size;
    pipe->param.cell_size     = cell_size;
    pipe->param.unit_size     = unit_size;
    pipe->param.unit_num      = unit_num;

    return 0;
}

void                 fip_pipe_cuda_free                      (fip_pipe_cuda_state*     pipe){
    free(pipe);
}

void                 fip_pipe_cuda_clear                     (fip_pipe_cuda_state**    pipe_ptr){
    if(pipe_ptr){
        fip_pipe_cuda_free(*pipe_ptr);
        *pipe_ptr = NULL;
    }
}

static cudaError_t   fip_pipe_cuda_select_device             (fip_pipe_cuda_state*     pipe){
    cudaError_t cudaError;

    if((cudaError = cudaSetDevice(pipe->device.ordinal))){
        fprintf(stderr, "Cannot set CUDA device: %s (%d)\n",
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

    if(pipe->param.verbose >= 10){
        printf("Selected GPU %d: %s (UUID: %s, PCIe %s)\n",
               pipe->device.ordinal,
               pipe->device.name,
               pipe->device.uuid,
               pipe->device.pci);
    }

    return cudaSuccess;
}

static cudaError_t   fip_pipe_cuda_plan_mem                  (fip_pipe_cuda_state*     pipe){
    cudaError_t cudaError;
    size_t      memfree=0, memtotal=0, memest=0;

    pipe->ring.depth.vis_bin       = 2;
    pipe->ring.depth.transform_gpu = 8;
    pipe->ring.depth.grid          = 3;
    pipe->ring.depth.image         = 5;
    pipe->ring.depth.result        = 2;

    if((cudaError = cudaMemGetInfo(&memfree, &memtotal))){
        fprintf(stderr, "Cannot query free memory on selected device! %s (%d)\n",
                        cudaGetErrorString(cudaError), (int)cudaError);
        fflush (stderr);
        return cudaError;
    }

    memest = pipe->ring.depth.vis_bin       * pipe->param.num_baselines     * sizeof(cuComplex) +  /* Visibilities */
             pipe->ring.depth.vis_bin       * pipe->param.num_baselines * 2 * sizeof(float)     +  /* Coordinates */
             pipe->ring.depth.transform_gpu * 3                         * 3 * sizeof(float)     +  /* Transform */
             pipe->ring.depth.grid          * pipe->param.grid_size     *
                                              pipe->param.grid_size     *     sizeof(cuComplex) +  /* Grid */
             pipe->ring.depth.image         * pipe->param.image_size    *
                                              pipe->param.image_size    *     sizeof(float)     +  /* Image */
             pipe->ring.depth.result        * pipe->param.unit_num      *
                                              pipe->param.unit_num      *     sizeof(float)     +  /* Result */
             pipe->ring.depth.image                                     *     sizeof(float);       /* Max */

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

static cudaError_t   fip_pipe_cuda_plan_launch               (fip_pipe_cuda_state*     pipe){
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

static cudaError_t   fip_pipe_cuda_destroy_events            (fip_pipe_cuda_state*     pipe){
    size_t t, i;

    cudaEventDestroy(pipe->events.pipestart);
    cudaEventDestroy(pipe->events.loopstart);
    cudaEventDestroy(pipe->events.loopend);
    cudaEventDestroy(pipe->events.pipeend);

    for(t=0; t<sizeof(pipe->events.iter) /
               sizeof(pipe->events.iter[0]); t++){
        for(i=ITER_START; i<ITER_NUM_EVENTS; i++){
            cudaEventDestroy(pipe->events.iter[t][i]);
        }
    }

    for(i=0; i<pipe->ring.depth.vis_bin; i++)
        cudaEventDestroy(pipe->events.ring.vis_pinned[i]);
    for(i=0; i<pipe->ring.depth.vis_bin; i++)
        cudaEventDestroy(pipe->events.ring.vis_gpu[i]);
    for(i=0; i<pipe->ring.depth.transform_gpu; i++)
        cudaEventDestroy(pipe->events.ring.transform_gpu[i]);
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
    free(pipe->events.ring.transform_gpu);
    free(pipe->events.ring.grid_gpu);
    free(pipe->events.ring.image_gpu);
    free(pipe->events.ring.result_gpu);
    free(pipe->events.ring.result_pinned);

    pipe->events.ring.vis_pinned    = NULL;
    pipe->events.ring.vis_gpu       = NULL;
    pipe->events.ring.transform_gpu = NULL;
    pipe->events.ring.grid_gpu      = NULL;
    pipe->events.ring.image_gpu     = NULL;
    pipe->events.ring.result_gpu    = NULL;
    pipe->events.ring.result_pinned = NULL;

    return cudaSuccess;
}

static cudaError_t   fip_pipe_cuda_create_events             (fip_pipe_cuda_state*     pipe){
    size_t t, i;

    pipe->events.ring.vis_pinned    = calloc(pipe->ring.depth.vis_bin,       sizeof(cudaEvent_t));
    pipe->events.ring.vis_gpu       = calloc(pipe->ring.depth.vis_bin,       sizeof(cudaEvent_t));
    pipe->events.ring.transform_gpu = calloc(pipe->ring.depth.transform_gpu, sizeof(cudaEvent_t));
    pipe->events.ring.grid_gpu      = calloc(pipe->ring.depth.grid,          sizeof(cudaEvent_t));
    pipe->events.ring.image_gpu     = calloc(pipe->ring.depth.image,         sizeof(cudaEvent_t));
    pipe->events.ring.result_gpu    = calloc(pipe->ring.depth.result,        sizeof(cudaEvent_t));
    pipe->events.ring.result_pinned = calloc(pipe->ring.depth.result,        sizeof(cudaEvent_t));

    if(!pipe->events.ring.vis_pinned    ||
       !pipe->events.ring.vis_gpu       ||
       !pipe->events.ring.transform_gpu ||
       !pipe->events.ring.grid_gpu      ||
       !pipe->events.ring.image_gpu     ||
       !pipe->events.ring.result_gpu    ||
       !pipe->events.ring.result_pinned){
        free(pipe->events.ring.vis_pinned);
        free(pipe->events.ring.vis_gpu);
        free(pipe->events.ring.transform_gpu);
        free(pipe->events.ring.grid_gpu);
        free(pipe->events.ring.image_gpu);
        free(pipe->events.ring.result_gpu);
        free(pipe->events.ring.result_pinned);

        pipe->events.ring.vis_pinned    = NULL;
        pipe->events.ring.vis_gpu       = NULL;
        pipe->events.ring.transform_gpu = NULL;
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
        for(i=ITER_START; i<ITER_NUM_EVENTS; i++){
            cudaEventCreate(&pipe->events.iter[t][i]);
        }
    }

    for(i=0; i<pipe->ring.depth.vis_bin; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.vis_pinned[i],    cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.vis_pinned[i],    0);
    }
    for(i=0; i<pipe->ring.depth.vis_bin; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.vis_gpu[i],       cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.vis_gpu[i],       0);
    }
    for(i=0; i<pipe->ring.depth.transform_gpu; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.transform_gpu[i], cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.transform_gpu[i], 0);
    }
    for(i=0; i<pipe->ring.depth.grid; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.grid_gpu[i],      cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.grid_gpu[i],      0);
    }
    for(i=0; i<pipe->ring.depth.image; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.image_gpu[i],     cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.image_gpu[i],     0);
    }
    for(i=0; i<pipe->ring.depth.result; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.result_gpu[i],    cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.result_gpu[i],    0);
    }
    for(i=0; i<pipe->ring.depth.result; i++){
        cudaEventCreateWithFlags(&pipe->events.ring.result_pinned[i], cudaEventDisableTiming);
        cudaEventRecord         ( pipe->events.ring.result_pinned[i], 0);
    }

    return cudaSuccess;
}

static cudaError_t   fip_pipe_cuda_record                    (fip_pipe_cuda_state*     pipe,
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

static cudaError_t   fip_pipe_cuda_await                     (fip_pipe_cuda_state*     pipe,
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

static cudaError_t   fip_pipe_cuda_lock                      (fip_pipe_cuda_state*     pipe,
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
        case RING_XFORM_GPU:
            iter %= pipe->ring.depth.transform_gpu;
            return cudaStreamWaitEvent(stream, pipe->events.ring.transform_gpu[iter], flags);
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

static cudaError_t   fip_pipe_cuda_unlock                    (fip_pipe_cuda_state*     pipe,
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
        case RING_XFORM_GPU:
            iter %= pipe->ring.depth.transform_gpu;
            return cudaEventRecordWithFlags(pipe->events.ring.transform_gpu[iter], stream, flags);
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

static cudaError_t   fip_pipe_cuda_plan_npp                  (fip_pipe_cuda_state*     pipe,
                                                              NppiSize*                npp_image_size,
                                                              NppStreamContext*        npp_ctx){
    /**
     * CUDA 12.4+ (NPP 12.2.5.2+) changed the data type for workspace sizes
     * from int to size_t.
     *
     * Use a temporary variable of the appropriate type to receive the result,
     * then promote to size_t.
     */

#if (NPP_VERSION_MAJOR  > 12) || \
    (NPP_VERSION_MAJOR == 12  && NPP_VERSION_MINOR  > 2) || \
    (NPP_VERSION_MAJOR == 12  && NPP_VERSION_MINOR == 2  && NPP_VERSION_PATCH >= 5) || \
    (NPP_VERSION_MAJOR == 12  && NPP_VERSION_MINOR == 2  && NPP_VERSION_PATCH == 5  && NPP_VERSION_BUILD >= 2)
    size_t maxsz = 0;
#else
    int    maxsz = 0;
#endif

    npp_image_size->height = (int)pipe->param.image_size;
    npp_image_size->width  = (int)pipe->param.image_size;
    nppGetStreamContext(npp_ctx);
    npp_ctx->hStream       = pipe->stream.gridding;
    cudaStreamGetFlags(npp_ctx->hStream, &npp_ctx->nStreamFlags);
    nppiMaxGetBufferHostSize_32f_C1R_Ctx(*npp_image_size, &maxsz, *npp_ctx);
    pipe->wrkspc.npp.sz = (size_t)maxsz;
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

static void          fip_pipe_cuda_dump_timings              (fip_pipe_cuda_state*     pipe){
    float milliseconds;

    if(pipe->param.verbose < 20)
        return;

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
}

static cudaError_t   fip_pipe_cuda_free_mem                  (fip_pipe_cuda_state*     pipe){
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

static cudaError_t   fip_pipe_cuda_alloc_mem                 (fip_pipe_cuda_state*     pipe){
    cudaError_t cudaError;

    size_t num_baselines       = pipe->param.num_baselines;
    size_t grid_size           = pipe->param.grid_size;
    size_t image_size          = pipe->param.image_size;
    size_t unit_num            = pipe->param.unit_num;

    cudaMallocPitch(&pipe->ring.vis_bin_gpu,
                    &pipe->ring.stride.vis_bin,
                    num_baselines * sizeof(float),
                    pipe->ring.depth.vis_bin * (2+2));
    cudaMallocPitch(&pipe->ring.grid_gpu,
                    &pipe->ring.stride.grid,
                    grid_size     * sizeof(cuComplex),
                    pipe->ring.depth.grid    * grid_size);
    cudaMallocPitch(&pipe->ring.image_gpu,
                    &pipe->ring.stride.image,
                    image_size    * sizeof(float),
                    pipe->ring.depth.image   * image_size);
    cudaMallocPitch(&pipe->ring.result_gpu,
                    &pipe->ring.stride.result,
                    unit_num      * sizeof(float),
                    pipe->ring.depth.result  * unit_num);

    pipe->ring.stride.vis_bin        /= sizeof(float);
    pipe->ring.stride.grid           /= sizeof(cuComplex);
    pipe->ring.stride.image          /= sizeof(float);
    pipe->ring.stride.result         /= sizeof(float);

    /**
     * We do things a bit differently for transform_gpu, given that it is
     * a fairly small, 3x3 matrix. We decide on the stride first (for
     * efficiency, round this up to the next natural power of 2 (16), and
     * allocate the requisite multiple of that memory using cudaMalloc().
     */

    pipe->ring.stride.transform_gpu = 16;
    cudaMalloc     (&pipe->ring.transform_gpu,
                     pipe->ring.depth.transform_gpu  *
                     pipe->ring.stride.transform_gpu * sizeof(float));

    cudaMalloc     (&pipe->wrkspc.npp.ptr,              pipe->wrkspc.npp.sz);
    cudaMalloc     (&pipe->wrkspc.cufft.ptr,            pipe->wrkspc.cufft.sz);
    cudaMalloc     (&pipe->wrkspc.conv_corr_kernel,     (image_size/2+1)                                 * sizeof(float));
    cudaMalloc     (&pipe->ring.max_gpu,                pipe->ring.depth.image                           * sizeof(float));

    cudaMallocHost (&pipe->ring.vis_bin_pinned,         pipe->ring.depth.vis_bin * num_baselines * (2+2) * sizeof(float));
    cudaMallocHost (&pipe->ring.transform_pinned,       pipe->ring.depth.vis_bin *        3  *         3 * sizeof(float));
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
    i %= pipe->ring.depth.vis_bin;
    return (float(*)[3])((float*)pipe->ring.transform_pinned + 3*3*i);
}

static float       (*fip_pipe_cuda_calc_ring_transform_gpu   (fip_pipe_cuda_state*     pipe, size_t i))[3]{
    i %= pipe->ring.depth.transform_gpu;
    return (float(*)[3])((float*)pipe->ring.transform_gpu +
                                 pipe->ring.stride.transform_gpu * i);
}

static const float (*fip_pipe_cuda_calc_ring_transform_gpu_c (fip_pipe_cuda_state*     pipe, size_t i))[3]{
    return (const float(*)[3])fip_pipe_cuda_calc_ring_transform_gpu(pipe, i);
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
    fip_cuda_kernel_convkernel  (pipe->launch.Bk,
                                 pipe->launch.Tk, 0,
                                 pipe->stream.interpolation,
                                 pipe->wrkspc.conv_corr_kernel,
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
        fip_pipe_cuda_lock      (pipe, i, RING_XFORM_GPU,  pipe->stream.copy_gpu, 0);
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
        fip_cuda_kernel_gridding(pipe->launch.Bg,
                                 pipe->launch.Tg, 0,
                                 pipe->stream.gridding,
                                 fip_pipe_cuda_calc_ring_grid_gpu        (pipe, i),
                                 fip_pipe_cuda_calc_ring_vis_gpu         (pipe, i), // Vis_real, Vis_imag
                                 fip_pipe_cuda_calc_ring_coords_gpu      (pipe, i), // Bin
                                 fip_pipe_cuda_calc_ring_transform_gpu_c (pipe, i), // V
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
        fip_cuda_kernel_interp  (pipe->launch.Bs,
                                 pipe->launch.Ts, 0,
                                 pipe->stream.interpolation,
                                 fip_pipe_cuda_calc_ring_image_gpu       (pipe, i),
                                 fip_pipe_cuda_calc_ring_grid_gpu        (pipe, i),
                                 fip_pipe_cuda_calc_ring_transform_gpu_c (pipe, i),
                                 pipe->ring.stride.image,
                                 pipe->param.image_size,
                                 pipe->ring.stride.grid,
                                 pipe->param.grid_size,
                                 pipe->param.cell_size,
                                 pipe->wrkspc.conv_corr_kernel,
                                 conv_corr_norm_factor,
                                 inv_num_baselines);
        fip_pipe_cuda_unlock    (pipe, i, RING_XFORM_GPU,  pipe->stream.interpolation, 0);
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
        fip_cuda_kernel_tlisi   (pipe->launch.Bt,
                                 pipe->launch.Tt,
                                 pipe->launch.St,
                                 pipe->stream.tlisi,
                                 fip_pipe_cuda_calc_ring_result_gpu      (pipe, i),
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

    fip_pipe_cuda_free_mem      (pipe);
    fip_pipe_cuda_record        (pipe, 0, PIPE_END, 0, 0);
    fip_pipe_cuda_dump_timings  (pipe);
    fip_pipe_cuda_destroy_events(pipe);

    return 0;
}
