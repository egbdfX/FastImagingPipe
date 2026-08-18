/* Include Guard */
#ifndef INCLUDE_FIP_CUDA_PIPELINE_H
#define INCLUDE_FIP_CUDA_PIPELINE_H



/* Includes */
#include <stdlib.h>
#include <string.h>
#include "fip/visibility.h"



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Struct Forward Declarations */
struct fip_pipe_cuda_state;



/* Typedefs */
typedef struct fip_pipe_cuda_state fip_pipe_cuda_state;

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





/* Function Prototypes */
LIBFIP_PUBLIC
int  fip_pipe_cuda_alloc(fip_pipe_cuda_state**    pipe_ptr,
                         const int                verbose,
                         const int                gpu_ordinal,
                         const size_t             num_baselines,
                         const size_t             image_size,
                         const float              cell_size,
                         const size_t             unit_size,
                         const size_t             unit_num,
                         const int                big_endian);

LIBFIP_PUBLIC
int  fip_pipe_cuda      (fip_pipe_cuda_state*     pipe,
                         fip_pipe_cuda_input_cb   callback_input,
                         fip_pipe_cuda_output_cb  callback_output,
                         void*                    userdata0,
                         void*                    userdata1,
                         const size_t             snap_start,
                         const size_t             snap_end);

LIBFIP_PUBLIC
void fip_pipe_cuda_free (fip_pipe_cuda_state*     pipe);

LIBFIP_PUBLIC
void fip_pipe_cuda_clear(fip_pipe_cuda_state**    pipe_ptr);



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
