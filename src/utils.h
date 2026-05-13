/* Include Guard */
#ifndef SRC_UTILS_H
#define SRC_UTILS_H



/* Includes */
#include <stdlib.h>



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Function Prototypes */
int FIpipe             (float* Visreal,
                        float* Visimag,
                        float* Bin,
                        float* Vin,
                        float* result_array,
                        size_t num_baselines,
                        size_t image_size,
                        size_t num_snapshots,
                        float  cell_size,
                        size_t unit_size);
int FIpipe2            (float* Visreal,
                        float* Visimag,
                        float* Bin,
                        float* Vin,
                        float* result_array,
                        size_t num_baselines,
                        size_t image_size,
                        size_t num_snapshots,
                        float  cell_size,
                        size_t unit_size);
float* read_fits_image (const char* filename, long*  naxes);
int    write_fits_image(const char* filename, float* image_data, long* naxes);



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
