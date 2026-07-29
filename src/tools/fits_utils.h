/* Include Guard */
#ifndef SRC_TOOLS_FITS_UTILS_H
#define SRC_TOOLS_FITS_UTILS_H



/* Includes */
#include <stdlib.h>
#include <fitsio.h>



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Function Prototypes */
int    fip_input_open_diskfile   (fitsfile**  fptr,
                                  const char* filename,
                                  int         iomode,
                                  int*        status);

int    fip_input_get_stats       (fitsfile*   fptr,
                                  long long*  num_snapshots,
                                  long long*  num_baselines,
                                  int*        status);

int    fip_output_open_diskfile  (fitsfile**  fptr,
                                  const char* filename,
                                  int         iomode,
                                  long long   snap_count,
                                  long long   unit_num,
                                  int*        status);

int    fip_output_create_diskfile(fitsfile**  fptr,
                                  const char* filename,
                                  long long   snap_count,
                                  long long   unit_num,
                                  int*        status);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
