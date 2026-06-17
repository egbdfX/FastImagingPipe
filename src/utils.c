/* Includes */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "utils.h"


/* Defines */
#define FITS_RETURN(ret)                        \
    do{                                         \
        return status ? *status=(ret) : (ret);  \
    }while(0)

#define FITS_CHECKED(x)                         \
    do{                                         \
        int _statustmp = (x);                   \
        if(_statustmp){                         \
            FITS_RETURN((_statustmp));          \
        }                                       \
    }while(0)

#define FITS_ASSERT(expr, ret, ...)             \
    do{                                         \
        if(!(expr)){                            \
            fprintf(stderr, __VA_ARGS__);       \
            FITS_RETURN((ret));                 \
        }                                       \
    }while(0)


float* read_fits_image(const char* filename, long* naxes) {
    fitsfile *fptr;
    int status = 0;

    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(stderr, status);
        return NULL;
    }

    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    printf("DEBUG: Reading %s, naxis=%d\n", filename, naxis);

    // CRITICAL FIX: Handle 1D and 2D arrays differently
    long total_elements;
    if (naxis == 1) {
        // For 1D arrays, only get one dimension
        long temp_naxes[1];
        fits_get_img_size(fptr, 1, temp_naxes, &status);
        if (status) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return NULL;
        }
        naxes[0] = temp_naxes[0];
        naxes[1] = 1;  // Force second dimension to 1 for consistency
        total_elements = naxes[0];
        printf("DEBUG: 1D array, naxes[0]=%ld, total_elements=%ld\n", naxes[0], total_elements);
    } else if (naxis == 2) {
        // For 2D arrays, get both dimensions
        fits_get_img_size(fptr, 2, naxes, &status);
        if (status) {
            fits_report_error(stderr, status);
            fits_close_file(fptr, &status);
            return NULL;
        }
        total_elements = naxes[0] * naxes[1];
        printf("DEBUG: 2D array, naxes[0]=%ld, naxes[1]=%ld, total_elements=%ld\n",
               naxes[0], naxes[1], total_elements);
    } else {
        fprintf(stderr, "ERROR: Unsupported naxis=%d for file %s\n", naxis, filename);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Allocate based on actual total elements
    float *image_data = (float *)malloc(total_elements * sizeof(float));
    if (image_data == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed for %ld elements (%.2f MB)\n",
                total_elements, (total_elements * sizeof(float)) / (1024.0 * 1024.0));
        fits_close_file(fptr, &status);
        return NULL;
    }

    printf("DEBUG: Allocated %.2f MB for %ld elements\n",
           (total_elements * sizeof(float)) / (1024.0 * 1024.0), total_elements);

    // Read the actual number of elements
    fits_read_img(fptr, TFLOAT, 1, total_elements, NULL, image_data, NULL, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        fits_close_file(fptr, &status);
        return NULL;
    }

    printf("DEBUG: Successfully read %ld elements, first value=%.6e, last value=%.6e\n",
           total_elements, image_data[0], image_data[total_elements-1]);

    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        return NULL;
    }

    return image_data;
}

int write_fits_image(const char* filename, float *image_data, long* naxes) {
    fitsfile *fptr;
    int status = 0;

    fitsfile *tmp_fptr;
    fits_open_file(&tmp_fptr, filename, READWRITE, &status);
    if (!status) {
        fits_delete_file(tmp_fptr, &status);
    }
    status = 0;

    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    long naxis = 2;
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    fits_write_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], image_data, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    return 0;
}


/**
 * @brief Compute number of extra header records required.
 * @param [in] datastart  The address of the start of the HDU's data, in bytes.
 * @return The number of extra blocks required in the HDU header prior to the
 *         start of that HDU's data. These extra blocks must be present, but may
 *         be filled with blanks if required.
 */

size_t fip_compute_missing_records(size_t datastart){
    return (datastart >> 6) * -37U & 63;
}

/**
 * @brief Open FIP input file.
 *
 * Identical interface to fits_open_diskfile() except executes additional
 * FIP-specific format checks.
 */

int fip_input_open_diskfile(fitsfile** fptr, const char* filename, int iomode, int* status){
    int       exttype    =  0;
    int       naxis      =  0;
    long long visaxis[3] = {0};
    long long binaxis[3] = {0};
    long long vinaxis[3] = {0};
    int       visdtype   =  0;
    int       bindtype   =  0;
    long long visdstart  = -1;
    long long bindstart  = -1;


    /* Open File */
    FITS_CHECKED(fits_open_diskfile (fptr,     filename,   iomode,  status));
    FITS_CHECKED(fits_movabs_hdu    (*fptr,    1,  &exttype,        status));


    /* Primary HDU (index=1 in 1-based indexing) is ignored. */


    /* First Extension HDU (index=2) must be the transform matrices. */
    FITS_CHECKED(fits_movrel_hdu    (*fptr,   1,  &exttype,         status));
    FITS_ASSERT (exttype == IMAGE_HDU,     NOT_IMAGE,
        "HDU 2 (transform matrices) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (*fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,               BAD_NAXIS,
        "HDU 2 (transform matrices) of file %s is supposed to be shaped "
        "{num_snapshots, 3, 3} (in C-order), but NAXIS was %d, thus "
        "cannot be a properly constructed FIP input file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(*fptr,   3,   vinaxis,         status));
    FITS_ASSERT (vinaxis[0] == 3,          BAD_NAXES,
        "HDU 2 (transform matrices) of file %s is supposed to be shaped "
        "{num_snapshots, 3, 3} (in C-order), but NAXIS1 was %lld, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename, vinaxis[0]);
    FITS_ASSERT (vinaxis[1] == 3,          BAD_NAXES,
        "HDU 2 (transform matrices) of file %s is supposed to be shaped "
        "{num_snapshots, 3, 3} (in C-order), but NAXIS2 was %lld, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename, vinaxis[1]);


    /* Second Extension HDU (index=3) must be the visibilities. */
    FITS_CHECKED(fits_movrel_hdu    (*fptr,   1,  &exttype,         status));
    FITS_ASSERT (exttype == IMAGE_HDU, NOT_IMAGE,
        "HDU 3 (visibilities) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (*fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,           BAD_NAXIS,
        "HDU 3 (visibilities) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS "
        "was %d, thus cannot be a properly constructed FIP input file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(*fptr,   3,   visaxis,         status));
    FITS_ASSERT (visaxis[0] == 2,          BAD_NAXES,
        "HDU 3 (visibilities) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS1 "
        "was %lld, thus cannot be a properly constructed FIP input file!\n",
        filename, visaxis[0]);
    FITS_ASSERT (vinaxis[2] == visaxis[2], BAD_NAXES,
        "HDU 3 (visibilities) of file %s is supposed to match HDU 2 "
        "(transform matrices) in num_snapshots, but their NAXIS3 values "
        "disagree (%lld != %lld); thus this file cannot be a properly "
        "constructed FIP input file!\n",
        filename, vinaxis[2], visaxis[2]);
    FITS_CHECKED(fits_get_img_type  (*fptr,       &visdtype,        status));
    FITS_ASSERT (visdtype == FLOAT_IMG,    BAD_BITPIX,
        "HDU 3 (visibilities) of file %s is supposed to be FLOAT32, "
        "but is not; thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_hduaddrll (*fptr, NULL, &visdstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(visdstart), BAD_HEADER_FILL,
        "HDU 3 (visibilities) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP input file!\n",
        filename, fip_compute_missing_records(visdstart));


    /* Third Extension HDU (index=4) must be the (r-)coordinates. */
    FITS_CHECKED(fits_movrel_hdu    (*fptr,   1,  &exttype,         status));
    FITS_ASSERT (exttype == IMAGE_HDU,     NOT_IMAGE,
        "HDU 4 (r-coordinates) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (*fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,               BAD_NAXIS,
        "HDU 4 (r-coordinates) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS "
        "was %d, thus cannot be a properly constructed FIP input file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(*fptr,   3,   binaxis,         status));
    FITS_ASSERT (binaxis[0] == 2,          BAD_NAXES,
        "HDU 4 (r-coordinates) of file %s is supposed to be shaped "
        "{num_snapshots, num_baselines, 2} (in C-order), but NAXIS1 "
        "was %lld, thus cannot be a properly constructed FIP input file!\n",
        filename, binaxis[0]);
    FITS_ASSERT (binaxis[1] == visaxis[1], BAD_NAXES,
        "HDU 4 (r-coordinates) of file %s is supposed to match HDU 3 "
        "(visibilities) in num_baselines, but their NAXIS2 values disagree "
        "(%lld != %lld); thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename, binaxis[1], visaxis[1]);
    FITS_ASSERT (binaxis[2] == visaxis[2], BAD_NAXES,
        "HDU 4 (r-coordinates) of file %s is supposed to match HDU 2 "
        "(visibilities) in num_snapshots, but their NAXIS3 values disagree "
        "(%lld != %lld); thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename, binaxis[2], visaxis[2]);
    FITS_CHECKED(fits_get_img_type  (*fptr,       &bindtype,        status));
    FITS_ASSERT (bindtype == FLOAT_IMG,    BAD_BITPIX,
        "HDU 4 (r-coordinates) of file %s is supposed to be FLOAT32, "
        "but is not; thus this file cannot be a properly constructed "
        "FIP input file!\n",
        filename);
    FITS_CHECKED(fits_get_hduaddrll (*fptr, NULL, &bindstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(bindstart), BAD_HEADER_FILL,
        "HDU 4 (r-coordinates) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP input file!\n",
        filename, fip_compute_missing_records(bindstart));


    /* Rewind to beginning of file */
    FITS_CHECKED(fits_movabs_hdu    (*fptr,   1,  &exttype,         status));


    /* Exit */
    FITS_RETURN(0);
}

/**
 * @brief Get statistics from FIP input file.
 * @param [in]   fptr               FITS file pointer.
 * @param [out]  num_snapshots      Number of snapshots.
 * @param [out]  num_baselines      Number of baselines.
 * @param [out]  status             FITS status code return.
 * @return 0 if successful, !0 otherwise.
 */

int fip_input_get_stats(fitsfile *fptr, long long* num_snapshots, long long* num_baselines, int* status){
    long long visaxis[3] = {0};

    *num_snapshots = 0;
    *num_baselines = 0;

    /**
     * Consult HDU 3 (visibilities), which should have
     *     NAXIS=3
     *     NAXIS1=2
     *     NAXIS2=num_baselines
     *     NAXIS3=num_snapshots
     * Then rewind to beginning of file.
     */

    FITS_CHECKED(fits_movabs_hdu    (fptr, 3, NULL,    status));
    FITS_CHECKED(fits_get_img_sizell(fptr, 3, visaxis, status));
    FITS_CHECKED(fits_movabs_hdu    (fptr, 1, NULL,    status));

    if(visaxis[0] != 2 || visaxis[1] <= 0 || visaxis[2] <= 0)
        FITS_RETURN(BAD_NAXES);

    *num_baselines = visaxis[1];
    *num_snapshots = visaxis[2];


    /* Exit */
    FITS_RETURN(0);
}

/**
 * @brief Open FIP output file.
 *
 * Similar interface to fits_open_diskfile(), except executes additional
 * FIP-specific format checks.
 *
 * @param [out]  fptr         FITS file pointer.
 * @param [in]   filename     Path to file to open.
 * @param [in]   iomode       I/O mode. Should almost certainly be READWRITE.
 * @param [out]  snap_count   Expected number of snapshots.
 * @param [out]  unit_num     Expected number of units per image.
 * @param [out]  status       FITS status code return.
 * @return 0 if successful, !0 otherwise.
 */

int fip_output_open_diskfile(fitsfile**  fptr,
                             const char* filename,
                             int         iomode,
                             long long   snap_count,
                             long long   unit_num,
                             int*        status){
    int       exttype    =  0;
    int       naxis      =  0;
    long long outaxis[3] = {0};
    int       outdtype   =  0;
    long long outdstart  = -1;


    /* Open File */
    FITS_CHECKED(fits_open_diskfile (fptr,     filename,   iomode,  status));
    FITS_CHECKED(fits_movabs_hdu    (*fptr,    1, &exttype,         status));


    /* Primary HDU (index=1 in 1-based indexing) must be the output image array. */
    FITS_ASSERT (exttype == IMAGE_HDU,     NOT_IMAGE,
        "HDU 1 (output image array) of file %s is not an IMAGE_HDU, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename);
    FITS_CHECKED(fits_get_img_dim   (*fptr,       &naxis,           status));
    FITS_ASSERT (naxis == 3,               BAD_NAXIS,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS was %d, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, naxis);
    FITS_CHECKED(fits_get_img_sizell(*fptr,   3,   outaxis,         status));
    FITS_ASSERT (outaxis[0] == unit_num,   BAD_NAXES,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS1 was %lld, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, outaxis[0]);
    FITS_ASSERT (outaxis[1] == unit_num,   BAD_NAXES,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS2 was %lld, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, outaxis[1]);
    FITS_ASSERT (outaxis[2] == snap_count, BAD_NAXES,
        "HDU 1 (output image array) of file %s is supposed to be shaped "
        "{num_snapshots, unit_num, unit_num} (in C-order), but NAXIS3 was %lld, "
        "thus cannot be a properly constructed FIP output file!\n",
        filename, outaxis[2]);
    FITS_CHECKED(fits_get_img_type  (*fptr,       &outdtype,        status));
    FITS_ASSERT (outdtype == FLOAT_IMG,    BAD_BITPIX,
        "HDU 1 (output image array) of file %s is supposed to be FLOAT32, "
        "but is not; thus this file cannot be a properly constructed "
        "FIP output file!\n",
        filename);
    FITS_CHECKED(fits_get_hduaddrll (*fptr, NULL, &outdstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(outdstart), BAD_HEADER_FILL,
        "HDU 1 (output image array) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP output file!\n",
        filename, fip_compute_missing_records(outdstart));


    /* Exit */
    FITS_RETURN(0);
}

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

int fip_output_create_diskfile(fitsfile**  fptr,
                               const char* filename,
                               long long   snap_count,
                               long long   unit_num,
                               int*        status){
    long long  outaxis[3] = {unit_num, unit_num, snap_count};
    long long  outdstart  = -1;
    int        hdrlen     = 0;
    int        hdrextra;


    /* Create File */
    FITS_CHECKED(fits_create_diskfile(fptr,  filename,               status));


    /* Primary HDU (index=1 in 1-based indexing) must be the output image array. */
    FITS_CHECKED(fits_create_imgll   (*fptr, FLOAT_IMG, 3, outaxis,  status));
    FITS_CHECKED(fits_get_hdrpos     (*fptr, &hdrlen,   NULL,        status));
    hdrextra = 36*64 - hdrlen%(36*64) - 1;
    FITS_CHECKED(fits_set_hdrsize    (*fptr, hdrextra,               status));
    FITS_CHECKED(fits_get_hduaddrll  (*fptr, NULL, &outdstart, NULL, status));
    FITS_ASSERT (!fip_compute_missing_records(outdstart), BAD_HEADER_FILL,
        "HDU 1 (output image array) of file %s is missing %zu records of 2880 "
        "bytes in its header; thus this file cannot be a properly "
        "constructed FIP output file!\n",
        filename, fip_compute_missing_records(outdstart));


    /* Exit */
    FITS_RETURN(0);
}
