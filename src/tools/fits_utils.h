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

/**
 * @brief Write fully a buffer with repeated pwrite().
 *
 * @param [in] fd   File descriptor to write to.
 * @param [in] buf  Pointer to buffer to write out.
 * @param [in] len  Length  of buffer to write out.
 * @param [in] off  Offset into file at which to write.
 * @return Number of bytes written >= 0, or error code.
 */

ssize_t fip_pwrite_fully            (int         fd,
                                     const void* buf,
                                     size_t      len,
                                     off_t       off);

/**
 * @brief Compute number of extra header records required.
 * @param [in] datastart  The address of the start of the HDU's data, in bytes.
 * @return The number of extra blocks required in the HDU header prior to the
 *         start of that HDU's data. These extra blocks must be present, but may
 *         be filled with blanks if required.
 */

size_t  fip_compute_missing_records (size_t datastart);

/**
 * @brief Validate FIP output file.
 *
 * Executes FIP output-format-specific validation checks.
 *
 * @param [in]   fptr         FITS file pointer.
 * @param [in]   filename     Path to file to open.
 * @param [out]  snap_count   Expected number of snapshots.
 * @param [out]  unit_num     Expected number of units per image.
 * @param [out]  status       FITS status code return.
 * @return 0 if successful, !0 otherwise.
 */

int     fip_output_validate_diskfile(fitsfile*   fptr,
                                     const char* filename,
                                     long long   snap_count,
                                     long long   unit_num,
                                     int*        status);

/**
 * @brief Create FIP output file.
 *
 * Similar interface to fits_create_diskfile().
 *
 * @param [out]  fptr         FITS file pointer.
 * @param [in]   filename     Path to file to open.
 * @param [out]  snap_count   Expected number of snapshots.
 * @param [out]  unit_num     Expected number of units per image.
 * @param [out]  status       FITS status code return.
 * @return 0 if successful, !0 otherwise.
 */

int     fip_output_create_diskfile  (fitsfile**  fptr,
                                     const char* filename,
                                     long long   snap_count,
                                     long long   unit_num,
                                     int*        status);

/**
 * @brief Create FIP output file, with file descriptor.
 *
 * Similar interface to fits_output_openat_diskfile(), except that it defaults
 * dirfd to the conventional default, AT_FDCWD (the current working directory).
 *
 * @param [out]  fptr         FITS file pointer.
 * @param [in]   filename     Path to file to open.
 * @param [out]  snap_count   Expected number of snapshots.
 * @param [out]  unit_num     Expected number of units per image.
 * @param [out]  status       FITS status code return.
 * @return File descriptor >= 0 if successful, negative errno code otherwise.
 */

int     fip_output_open_diskfile    (fitsfile**  fptr,
                                     const char* filename,
                                     long long   snap_count,
                                     long long   unit_num,
                                     int*        status);

/**
 * @brief Create FIP output file, with file descriptor.
 *
 * Similar interface to fits_create_diskfile().
 *
 * @param [out]  fptr         FITS file pointer.
 * @param [in]   dirfd        Directory file descriptor.
 * @param [in]   filename     Path to file to open.
 * @param [out]  snap_count   Expected number of snapshots.
 * @param [out]  unit_num     Expected number of units per image.
 * @param [out]  status       FITS status code return.
 * @return File descriptor >= 0 if successful, negative errno code otherwise.
 */

int     fip_output_openat_diskfile  (fitsfile**  fptr,
                                     int         dirfd,
                                     const char* filename,
                                     long long   snap_count,
                                     long long   unit_num,
                                     int*        status);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
