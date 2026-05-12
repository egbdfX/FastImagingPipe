#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <popt.h>
#include <cfitsio/fitsio.h>

#include "utils.h"


#define  HELP_FLAG  1



/* Forward  */



/* fip pipe implementation. */
int main_pipe(int argc, char* argv[]) {
    long imasize[2];

    int rc               = EXIT_FAILURE;

    char* input_Visreal  = (char*)"Visreal0.fits";
    char* input_Visimag  = (char*)"Visimag0.fits";
    char* input_Bin      = (char*)"Bin0.fits";
    char* input_Vin      = (char*)"Vin0.fits";
    char* output_name    = (char*)"dirty0.fits";

    size_t image_size    = 4096;
    size_t num_baselines = 130816;
    float  cell_size     = 0.0000072722;
    size_t num_snapshots = 3;
    size_t unit_size     = 32;


    /**
     * Parse arguments for command fip pipe.
     */
    poptContext parser;
    struct poptOption fip_pipe_opts[] = {
        {"help",          'h', POPT_ARG_NONE,     NULL,           HELP_FLAG, "Print this help",     NULL},
        {"input-visreal",  0,  POPT_ARG_STRING,   &input_Visreal, 0,         "Visreal",             "FILE"},
        {"input-visimag",  0,  POPT_ARG_STRING,   &input_Visimag, 0,         "Visimag",             "FILE"},
        {"input-bin",      0,  POPT_ARG_STRING,   &input_Bin,     0,         "Bin",                 "FILE"},
        {"input-vin",      0,  POPT_ARG_STRING,   &input_Vin,     0,         "Vin",                 "FILE"},
        {"image-size",    's', POPT_ARG_LONGLONG, &image_size,    0,         "Image Size",          "N"},
        {"num-baselines", 'b', POPT_ARG_LONGLONG, &num_baselines, 0,         "Number of baselines", "N"},
        {"cell-size",     'C', POPT_ARG_FLOAT,    &cell_size,     0,         "Cell Size",           "F>0.0"},
        {"num-snapshots", 'S', POPT_ARG_LONGLONG, &num_snapshots, 0,         "Number of snapshots", "N"},
        {"unit-size",     'u', POPT_ARG_LONGLONG, &unit_size,     0,         "Unit Size",           "N"},
        {"output",        'o', POPT_ARG_STRING,   &output_name,   0,         "Output",              "FILE"},
        {NULL,             0,  POPT_ARG_NONE,     NULL,           0,         NULL,                  NULL},
    };
    struct poptAlias help_alias = {NULL, '?', 0, NULL};


    parser = poptGetContext("fip-pipe", argc, (const char**)argv, fip_pipe_opts, 0);
    if(!parser){
        fprintf(stderr, "Out of memory\n");
        goto poptfail;
    }
    if((rc = poptReadDefaultConfig(parser, 0))){
        fprintf(stderr, "Failed to parse popt configuration, error %s (%d)\n",
                        strerror(errno), errno);
        goto poptfail;
    }
    if(poptParseArgvString("--help", &help_alias.argc,
                                     &help_alias.argv)){
        fprintf(stderr, "Out of memory\n");
        goto poptfail;
    }
    if(poptAddAlias(parser, help_alias, 0)){
        free(help_alias.argv);
        fprintf(stderr, "Out of memory\n");
        goto poptfail;
    }
    do{
        switch((rc = poptGetNextOpt(parser))){
            case -1:        /* <end of arguments> */
                rc = EXIT_SUCCESS;
                break;
            case HELP_FLAG: /* --help */
                poptPrintHelp(parser, stdout, 0);
                poptFreeContext(parser);
                return EXIT_SUCCESS;
            default:
                fprintf(stderr, "%s: %s (%d)\n",
                        poptBadOption(parser, 0),
                        poptStrerror(rc), rc);
                break;
        }
    }while(rc > 0);
    poptfail:
    poptFreeContext(parser);
    if(rc != EXIT_SUCCESS)
        return rc;


    float *Visreal = read_fits_image(input_Visreal, imasize);
    if (Visreal == NULL) {
        return EXIT_FAILURE;
    }

    float *Visimag = read_fits_image(input_Visimag, imasize);
    if (Visimag == NULL) {
        return EXIT_FAILURE;
    }

    float *Bin     = read_fits_image(input_Bin,     imasize);
    if (Bin == NULL) {
        return EXIT_FAILURE;
    }

    float *Vin     = read_fits_image(input_Vin,     imasize);
    if (Vin == NULL) {
        return EXIT_FAILURE;
    }

    size_t unit_num = image_size/unit_size;
    float* result_array = (float*)malloc(unit_num*unit_num*sizeof(float));

    FIpipe(Visreal, Visimag, Bin, Vin, result_array, num_baselines, image_size, num_snapshots, cell_size, unit_size);

    long naxes[2] = {unit_size, unit_size};
    int status = write_fits_image(output_name, result_array, naxes);
    if (status) {
        fprintf(stderr, "Error writing FITS image\n");
        return EXIT_FAILURE;
    }

    return rc;
}
