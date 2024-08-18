#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"
#include <vector>
#include <iostream>
#include <chrono>

int FIpipe(float* Visreal, float* Visimag, float* Bin, float* Vin, float* dirty_image, size_t num_baselines, size_t image_size, float freq_hz, float uv_scale, float phase_ra, float phase_dec);

using namespace std;

float* read_fits_image(const char* filename, long* naxes) {
    fitsfile *fptr; // FITS file pointer
    int status = 0; // Status variable for FITSIO functions

    // Open the FITS file
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(stderr, status);
        return NULL;
    }

    // Get image dimensions
    int naxis;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Allocate memory for image data
    float *image_data = (float *)malloc(naxes[0] * naxes[1] * sizeof(float));
    if (image_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Read image data
    fits_read_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], NULL, image_data, NULL, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        fits_close_file(fptr, &status);
        return NULL;
    }

    // Close the FITS file
    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        free(image_data);
        return NULL;
    }

    return image_data;
}

// Function to write FITS image
int write_fits_image(const char* filename, float *image_data, long* naxes) {
    fitsfile *fptr; // FITS file pointer
    int status = 0; // Status variable for FITSIO functions

    // Create new FITS file
    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    // Create image extension
    long naxis = 2; // 2-dimensional image
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    // Write image data
    fits_write_img(fptr, TFLOAT, 1, naxes[0] * naxes[1], image_data, &status);
    if (status) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return status;
    }

    // Close the FITS file
    fits_close_file(fptr, &status);
    if (status) {
        fits_report_error(stderr, status);
        return status;
    }

    return 0; // Success
}

int main(int argc, char* argv[]) {
	// Visreal, Visimag, Bin, Vin, Min
    char input_Visreal[1000];
    char input_Visimag[1000];
    char input_Bin[1000];
    char input_Vin[1000];
    char output_name[1000];
    
    long imasize[2];
    sprintf(input_Visreal, "%s", argv[1]);
    float *Visreal = read_fits_image(input_Visreal, imasize);
    if (Visreal == NULL) {
        return 1;
    }
    
    sprintf(input_Visimag, "%s", argv[2]);
    float *Visimag = read_fits_image(input_Visimag, imasize);
    if (Visimag == NULL) {
        return 1;
    }
    
    sprintf(input_Bin, "%s", argv[3]);
    float *Bin = read_fits_image(input_Bin, imasize);
    if (Bin == NULL) {
        return 1;
    }
    
    sprintf(input_Vin, "%s", argv[4]);
    float *Vin = read_fits_image(input_Vin, imasize);
    if (Vin == NULL) {
        return 1;
    }
    
    size_t image_size = std::stoul(argv[5]);
    float* dirty_image = (float*)malloc(image_size*image_size*sizeof(float));

    size_t num_baselines = std::stoul(argv[6]);
    float freq_hz = std::stof(argv[7]);
    float uv_scale = std::stof(argv[8]);
	float phase_ra = std::stof(argv[9]);
	float phase_dec = std::stof(argv[10]);
	
	sprintf(output_name, "%s", argv[11]);
	
	FIpipe(Visreal, Visimag, Bin, Vin, dirty_image, num_baselines, image_size, freq_hz, uv_scale, phase_ra, phase_dec);
	
	// Write FITS image
	long naxes[2] = {long(image_size), long(image_size)};
    int status = write_fits_image(output_name, dirty_image, naxes);
    if (status) {
        fprintf(stderr, "Error writing FITS image\n");
        return 1;
    }
}