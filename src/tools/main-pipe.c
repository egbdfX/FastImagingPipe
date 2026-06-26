#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <popt.h>
#include <fitsio.h>

#include "utils.h"
#include "fip-pipeline-cuda-state.h"


#define  HELP_FLAG               1
#define  START_OFFSET_FLAG       2
#define  END_OFFSET_FLAG         3
#define  SNAP_COUNT_GE_3_FLAG    4
#define  NUM_BASELINES_FLAG      5

#define  NUM_BASELINES_FLAG_DEFAULT      -1
#define  START_OFFSET_FLAG_DEFAULT       -1
#define  END_OFFSET_FLAG_DEFAULT          0
#define  SNAP_COUNT_FLAG_DEFAULT         -1



/* fip pipe implementation. */
int main_oldpipe(int argc, char* argv[]) {
    long imasize[2];

    int rc               = EXIT_FAILURE;

    char* input_Visreal  = (char*)"Visreal.fits";
    char* input_Visimag  = (char*)"Visimag.fits";
    char* input_Bin      = (char*)"Bin.fits";
    char* input_Vin      = (char*)"Vin.fits";
    char* output_name    = (char*)"dirty.fits";

    size_t image_size    = 1024;
    size_t num_baselines = 1812480;
    float  cell_size     = 0.000020595;
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

    FIpipe2(Visreal, Visimag, Bin, Vin, result_array, num_baselines, image_size, num_snapshots, cell_size, unit_size);

    long naxes[2] = {unit_num, unit_num};
    int status = write_fits_image(output_name, result_array, naxes);
    if (status) {
        fprintf(stderr, "Error writing FITS image\n");
        return EXIT_FAILURE;
    }

    return rc;
}

int main_pipe(int argc, char* argv[]){
    int       rc              = EXIT_FAILURE;
    int       fits_status     = 0;

    fitsfile* input           = NULL;
    char*     input_name      = (char*)"input.fits";
    fitsfile* output          = NULL;
    char*     output_name     = (char*)"output.fits";


    long long image_size      = 1024;
    long long unit_size       = 32;
    long long num_baselines   = NUM_BASELINES_FLAG_DEFAULT;
    long long snap_start      = START_OFFSET_FLAG_DEFAULT;
    long long snap_end        = END_OFFSET_FLAG_DEFAULT;
    long long snap_count      = SNAP_COUNT_FLAG_DEFAULT;
    float     cell_size       = 0.000020595;
    long long snap_count_file    = 0;
    long long num_baselines_file = 0;
    long long snap_start_final   = 0;
    long long snap_end_final     = 0;
    long long snap_count_final   = 0;


    /**
     * Parse arguments for command fip pipe.
     */

    poptContext parser;
    struct poptOption fip_pipe_opts[] = {
        {"help",          'h', POPT_ARG_NONE,     NULL,           HELP_FLAG,             "Print this help",            NULL},
        {"input",         'i', POPT_ARG_STRING,   &input_name,    0,                     "Input",                      "FILE"},
        {"output",        'o', POPT_ARG_STRING,   &output_name,   0,                     "Output",                     "FILE"},
        {"snap-count",    'N', POPT_ARG_LONGLONG, &snap_count,    SNAP_COUNT_GE_3_FLAG,  "Number of snapshots",        "N"},
        {"snap-start",    'S', POPT_ARG_LONGLONG, &snap_start,    START_OFFSET_FLAG,     "Starting snapshot# (incl.)", "N"},
        {"snap-end",      'E', POPT_ARG_LONGLONG, &snap_end,      END_OFFSET_FLAG,       "Ending snapshot#   (excl.)", "N"},
        {"image-size",    's', POPT_ARG_LONGLONG, &image_size,    0,                     "Image Size",                 "N"},
        {"num-baselines", 'b', POPT_ARG_LONGLONG, &num_baselines, 0,                     "Number of baselines",        "N"},
        {"cell-size",     'C', POPT_ARG_FLOAT,    &cell_size,     0,                     "Cell Size",                  "F>0.0"},
        {"unit-size",     'u', POPT_ARG_LONGLONG, &unit_size,     0,                     "Unit Size",                  "N"},
        {NULL,             0,  POPT_ARG_NONE,     NULL,           0,                     NULL,                         NULL},
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
            case HELP_FLAG:              /* --help */
                poptPrintHelp(parser, stdout, 0);
                poptFreeContext(parser);
                return EXIT_SUCCESS;
            case START_OFFSET_FLAG:      /* --snap-start=N */
                if(snap_start < 0 && snap_start > -3){
                    fprintf(stderr, "Argument --snap-start=%lld must be >= 0!\n", snap_start);
                    rc = EXIT_FAILURE;
                    goto poptfail;
                }
                break;
            case END_OFFSET_FLAG:        /* --snap-end=N */
                if(snap_end >= 0 && snap_end < 3){
                    fprintf(stderr, "Argument --snap-end=%lld must be >= 3!\n", snap_end);
                    rc = EXIT_FAILURE;
                    goto poptfail;
                }
                break;
            case SNAP_COUNT_GE_3_FLAG:   /* --snap-count=N */
                if(snap_count < 3){
                    fprintf(stderr, "Argument --snap-count=%lld must be >= 3!\n", snap_count);
                    rc = EXIT_FAILURE;
                    goto poptfail;
                }
                break;
            case NUM_BASELINES_FLAG:     /* --num-baselines=N */
                if(num_baselines <= 0){
                    fprintf(stderr, "Argument --num-baselines=%lld must be >= 1!\n", num_baselines);
                    rc = EXIT_FAILURE;
                    goto poptfail;
                }
                break;
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
    rc = EXIT_FAILURE;


    /**
     * Open input file.
     * 
     * Also collect the file's vital statistics, enabling its validation and
     * that of the program's other arguments.
     */

    if(fip_input_open_diskfile(&input,  input_name,       READONLY,           &fits_status) ||
       fip_input_get_stats    ( input, &snap_count_file, &num_baselines_file, &fits_status))
        goto fitsfail;

    if(snap_count_file < 3){
        fprintf(stderr, "Input file %s has %lld<3 snapshots!\n", input_name, snap_count_file);
        goto fitsfail;
    }
    if(num_baselines_file <= 0){
        fprintf(stderr, "Input file %s has no baselines!\n", input_name);
        goto fitsfail;
    }


    /**
     * Check arguments for sanity, now that we know the problem size from the input file.
     *
     * Start with
     *   * Single argument --num-baselines.
     *   * The triple set of arguments --snap-start/--snap-end/--snap-count.
     *     For this triple, handle every possibility:
     *       * None of them set (Select whole file)
     *       * One  of them set (Infer the other two from the size of the file)
     *       * Two  of them set (Infer the other one from the size of the file, if legal)
     *       * All three set    (Check consistency)
     */

    if      (num_baselines == NUM_BASELINES_FLAG_DEFAULT){
             num_baselines  = num_baselines_file;
    }else if(num_baselines  > num_baselines_file){
        fprintf(stderr, "Argument --num-baselines=%lld higher than file's contents %lld!\n",
                num_baselines, num_baselines_file);
        goto fitsfail;
    }

    /* Finalize the explicit arguments. */
    if(snap_start != START_OFFSET_FLAG_DEFAULT){
        if(snap_start >= 0){
            if(snap_start >= snap_count_file){
                fprintf(stderr, "Argument --snap-start=%lld implies a start "
                                "after the end of file %s!\n",
                        snap_start, input_name);
                goto fitsfail;
            }
            snap_start_final = snap_start;
        }else{
            if(snap_start < -snap_count_file){
                fprintf(stderr, "Argument --snap-start=%lld implies a start "
                                "before the beginning of file %s!\n",
                        snap_start, input_name);
                goto fitsfail;
            }
            snap_start_final = snap_count_file+snap_start;
        }
    }
    if(snap_end   != END_OFFSET_FLAG_DEFAULT){
        if(snap_end >= 0){
            if(snap_end > snap_count_file){
                fprintf(stderr, "Argument --snap-end=%lld implies an end "
                                "after the end of file %s!\n",
                        snap_end, input_name);
                goto fitsfail;
            }
            snap_end_final = snap_end;
        }else{
            if(snap_end <= -snap_count_file){
                fprintf(stderr, "Argument --snap-end=%lld implies an end "
                                "at or before the beginning of file %s!\n",
                        snap_end, input_name);
                goto fitsfail;
            }
            snap_end_final = snap_count_file+snap_end;
        }
    }
    if(snap_count != SNAP_COUNT_FLAG_DEFAULT){
        if(snap_count < 3){
            fprintf(stderr, "Argument --snap-count=%lld is less than 3!\n",
                    snap_count);
            goto fitsfail;
        }
        if(snap_count > snap_count_file){
            fprintf(stderr, "Argument --snap-count=%lld exceeds the number of "
                            "snapshots in file %s!\n",
                    snap_count, input_name);
            goto fitsfail;
        }
        snap_count_final = snap_count;
    }

    /* Finalize the implicit arguments by handling the 0/1/2/3-arguments-set cases */
    if      (snap_start == START_OFFSET_FLAG_DEFAULT &&
             snap_end   == END_OFFSET_FLAG_DEFAULT   &&
             snap_count == SNAP_COUNT_FLAG_DEFAULT){
        snap_start_final = 0;
        snap_end_final   = snap_count_file;
        snap_count_final = snap_count_file;
    }else if(snap_start != START_OFFSET_FLAG_DEFAULT &&
             snap_end   == END_OFFSET_FLAG_DEFAULT   &&
             snap_count == SNAP_COUNT_FLAG_DEFAULT){
        snap_end_final   = snap_count_file;
        snap_count_final = snap_count_file - snap_start_final;
    }else if(snap_start == START_OFFSET_FLAG_DEFAULT &&
             snap_end   != END_OFFSET_FLAG_DEFAULT   &&
             snap_count == SNAP_COUNT_FLAG_DEFAULT){
        snap_start_final = 0;
        snap_count_final = snap_end_final;
    }else if(snap_start == START_OFFSET_FLAG_DEFAULT &&
             snap_end   == END_OFFSET_FLAG_DEFAULT   &&
             snap_count != SNAP_COUNT_FLAG_DEFAULT){
        snap_start_final = 0;
        snap_end_final   = snap_count_final;
    }else if(snap_start == START_OFFSET_FLAG_DEFAULT &&
             snap_end   != END_OFFSET_FLAG_DEFAULT   &&
             snap_count != SNAP_COUNT_FLAG_DEFAULT){
        if(snap_count_final > snap_end_final){
            fprintf(stderr, "Arguments --snap-count=%lld --snap-end=%lld imply a start "
                            "before the first snapshot!\n", snap_count, snap_end);
            goto fitsfail;
        }
        snap_start_final = snap_end_final-snap_count_final;
    }else if(snap_start != START_OFFSET_FLAG_DEFAULT &&
             snap_end   == END_OFFSET_FLAG_DEFAULT   &&
             snap_count != SNAP_COUNT_FLAG_DEFAULT){
        if(snap_start_final > snap_count_file-snap_count_final){
            fprintf(stderr, "Arguments --snap-start=%lld --snap-count=%lld imply an end "
                            "after the last snapshot!\n", snap_start, snap_count);
            goto fitsfail;
        }
        snap_end_final = snap_start_final+snap_count_final;
    }else if(snap_start != START_OFFSET_FLAG_DEFAULT &&
             snap_end   != END_OFFSET_FLAG_DEFAULT   &&
             snap_count == SNAP_COUNT_FLAG_DEFAULT){
        if(snap_start_final >= snap_end_final){
            fprintf(stderr, "Arguments --snap-start=%lld --snap-end=%lld imply zero or "
                            "negative number of snapshots!\n", snap_start, snap_end);
            goto fitsfail;
        }
        snap_count_final = snap_end_final - snap_start_final;
    }else{
        if(snap_start_final >= snap_end_final){
            fprintf(stderr, "Arguments --snap-start=%lld --snap-end=%lld imply zero or "
                            "negative number of snapshots!\n", snap_start, snap_end);
            goto fitsfail;
        }
        if(snap_count_final != snap_end_final-snap_start_final){
            fprintf(stderr, "Arguments --snap-start=%lld --snap-count=%lld --snap-end=%lld "
                            "are inconsistent!\n", snap_start, snap_count, snap_end);
            goto fitsfail;
        }
    }

    /* Handle insanities of the finalized parameters not caught earlier */
    if(snap_count_final < 3){
        fprintf(stderr, "Implied --snap-count=%lld is less than 3!\n", snap_count_final);
        goto fitsfail;
    }


    /**
     * Open or create output file.
     *
     * In the event of a partially-created, corrupt output file, the error codes
     * below can appear:
     *
     *   - END_OF_FILE:    If the file is completely empty (0 bytes)
     *   - UNKNOWN_REC:    If the file has an incomplete, invalid header written.
     *
     * Assume that we want to rewrite the file in that case.
     */

    const size_t unit_num = image_size/unit_size;
    switch(fip_output_open_diskfile(&output, output_name, READWRITE,
                                    snap_count_file, unit_num,
                                    &fits_status)){
        case 0:
            break;
        case END_OF_FILE:
        case UNKNOWN_REC:
            if(fits_delete_file(output, &fits_status))
                goto fitsfail;
            /* FALLTHROUGH */
        case FILE_NOT_OPENED:
            fits_status = 0;
            if(fip_output_create_diskfile(&output, output_name,
                                          snap_count_file, unit_num,
                                          &fits_status))
                goto fitsfail;
            break;
        default:
            goto fitsfail;
    }


    /* Execute Pipeline */
    fip_pipeline_cuda_state pipe;
    fip_pipe_cuda_init(&pipe, num_baselines, image_size, cell_size, unit_size, unit_num);
    rc = fip_pipe_cuda(&pipe, input, output, snap_start_final, snap_end_final);
    fits_flush_file(output, &fits_status);


    /* Clean up and exit */
    fitsfail:
    if(fits_status){
        fits_report_error(stderr, fits_status);
        rc = EXIT_FAILURE;
    }
    if(input)
        fits_close_file(input,  &fits_status), input  = NULL;
    if(output)
        fits_close_file(output, &fits_status), output = NULL;
    return rc;
}
