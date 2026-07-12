#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <popt.h>
#include <fitsio.h>
#include <fcntl.h>
#include <unistd.h>

#include "fits-utils.h"
#include "fip-pipe-cuda.h"


#define  HELP_FLAG               1
#define  START_OFFSET_FLAG       2
#define  END_OFFSET_FLAG         3
#define  SNAP_COUNT_GE_3_FLAG    4
#define  NUM_BASELINES_FLAG      5
#define  VERBOSE_FLAG            6
#define  QUIET_FLAG              7

#define  NUM_BASELINES_FLAG_DEFAULT      -1
#define  START_OFFSET_FLAG_DEFAULT       -1
#define  END_OFFSET_FLAG_DEFAULT          0
#define  SNAP_COUNT_FLAG_DEFAULT         -1



/* fip pipe implementation. */
typedef struct{
    struct{
        int       status;
        int       fd;
        long long transform_off;
        long long visibilities_off;
        long long coords_off;
    } input;
    struct{
        int       status;
        fitsfile* result;
    } output;
} fip_pipe_iter_state;

static ssize_t pread_reliable(int fd, void* buf, size_t count, off_t offset){
    ssize_t breadt;
    ssize_t bread = 0;
    char*   bufc  = buf;

    while(bread < (ssize_t)count){
        breadt = pread(fd, bufc+bread, count-(size_t)bread, offset+bread);
        if(breadt <= 0){
            return bread>0 ? bread : breadt;
        }else{
            bread += breadt;
        }
    }

    return bread;
}

static void input_cb (void*  userdata0,
                      void*  userdata1,
                      void*  visibilities,
                      float* coords,
                      float  transform[3][3],
                      size_t num_baselines,
                      size_t iter){
    uint32_t             x;
    size_t               i;
    size_t               transform_count    = 3 * 3;
    size_t               visibilities_count = 2 * num_baselines;
    size_t               coords_count       = 2 * num_baselines;
    size_t               transform_bytes    = transform_count    * sizeof(float);
    size_t               visibilities_bytes = visibilities_count * sizeof(float);
    size_t               coords_bytes       = coords_count       * sizeof(float);
    fip_pipe_iter_state* state              = (fip_pipe_iter_state*)userdata0;
    (void)userdata1;

    pread_reliable(state->input.fd, transform,     transform_bytes,
                   state->input.transform_off    + transform_bytes    * iter);
    pread_reliable(state->input.fd, visibilities,  visibilities_bytes,
                   state->input.visibilities_off + visibilities_bytes * iter);
    pread_reliable(state->input.fd, coords,        coords_bytes,
                   state->input.coords_off       + coords_bytes       * iter);

    for(i=0; i<transform_count; i++){
        memcpy(&x, &((float*)transform)[i],    sizeof(uint32_t)); x = __builtin_bswap32(x);
        memcpy(&((float*)transform)[i], &x,    sizeof(uint32_t));
    }
    for(i=0; i<visibilities_count; i++){
        memcpy(&x, &((float*)visibilities)[i], sizeof(uint32_t)); x = __builtin_bswap32(x);
        memcpy(&((float*)visibilities)[i], &x, sizeof(uint32_t));
    }
    for(i=0; i<coords_count; i++){
        memcpy(&x, &coords[i],                 sizeof(uint32_t)); x = __builtin_bswap32(x);
        memcpy(&coords[i], &x,                 sizeof(uint32_t));
    }

    if(state->input.status){
        fits_report_error(stderr, state->input.status);
        fflush(stderr);
    }
}

static void output_cb(void*  userdata0,
                      void*  userdata1,
                      void*  result,
                      size_t unit_num,
                      size_t iter){
    fip_pipe_iter_state* state = (fip_pipe_iter_state*)userdata0;
    (void)userdata1;


    /**
     * At the iteration with canonical input snapshot number i,
     * the output being written is number i-2.
     *
     * Snapshots 0 and 1 have no corresponding output because at least
     * three snapshots are required to calculate a result image.
     */

    long fres[3] = {1, 1,               iter+1-2};
    long lres[3] = {unit_num, unit_num, iter+1-2};

    fits_write_subset(state->output.result, TFLOAT, fres, lres, result, &state->output.status);

    if(state->output.status){
        fits_report_error(stderr, state->output.status);
        fflush(stderr);
    }
}


int main_pipe(int argc, char* argv[]){
    int       rc              = EXIT_FAILURE;
    int       fits_status     = 0;

    fitsfile* input           = NULL;
    char*     input_name      = (char*)"input.fits";
    fitsfile* output          = NULL;
    char*     output_name     = (char*)"output.fits";

    fip_pipe_cuda_state* pipe = NULL;
    fip_pipe_iter_state state = {{0, -1, 0, 0, 0}, {0, NULL}};


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
    int       verbose         = 0;
    int       gpu_ordinal     = 0;


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
        {"gpu",           'g', POPT_ARG_INT,      &gpu_ordinal,   0,                     "GPU ordinal",                "N"},
        {"verbose",       'v', POPT_ARG_NONE,     NULL,           VERBOSE_FLAG,          "Increase logging verbosity", NULL},
        {"quiet",         'q', POPT_ARG_NONE,     NULL,           QUIET_FLAG,            "Decrease logging verbosity", NULL},
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
            case VERBOSE_FLAG:           /* --verbose, -v */
                if(verbose <= INT_MAX-10)
                    verbose += 10;
                break;
            case QUIET_FLAG:             /* --quiet,   -q */
                if(verbose >= INT_MIN+10)
                    verbose -= 10;
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

    const size_t unit_num            = image_size/unit_size;
    const size_t snap_count_file_out = snap_count_file-2;
    switch(fip_output_open_diskfile(&output, output_name, READWRITE,
                                    snap_count_file_out, unit_num,
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
                                          snap_count_file_out, unit_num,
                                          &fits_status))
                goto fitsfail;
            break;
        default:
            goto fitsfail;
    }


    /* Execute Pipeline */
    fits_movrel_hdu   (input, +1,                                  NULL, &fits_status);
    fits_get_hduaddrll(input, NULL, &state.input.transform_off,    NULL, &fits_status);
    fits_movrel_hdu   (input, +1,                                  NULL, &fits_status);
    fits_get_hduaddrll(input, NULL, &state.input.visibilities_off, NULL, &fits_status);
    fits_movrel_hdu   (input, +1,                                  NULL, &fits_status);
    fits_get_hduaddrll(input, NULL, &state.input.coords_off,       NULL, &fits_status);
    state.input.fd      = open(input_name, O_RDONLY|O_NOCTTY|O_CLOEXEC, 0);
    state.input.status  = 0;
    state.output.status = 0;
    state.output.result = output;
    if(fits_status || state.input.fd<0)
        goto fitsfail;

    if(fip_pipe_cuda_alloc(&pipe, verbose, gpu_ordinal, num_baselines, image_size, cell_size, unit_size, unit_num))
        goto cudafail;
    rc = fip_pipe_cuda(pipe, input_cb, output_cb, &state, NULL,
                             snap_start_final, snap_end_final);
    fip_pipe_cuda_clear(&pipe);

    fits_flush_file(output, &fits_status);


    /* Clean up and exit */
    cudafail:
    fitsfail:
    if(fits_status){
        fits_report_error(stderr, fits_status);
    }
    if(state.input.fd>=0){
        close(state.input.fd);
    }else{
        rc = EXIT_FAILURE;
    }
    if(input)
        fits_close_file(input,  &fits_status), input  = NULL;
    if(output)
        fits_close_file(output, &fits_status), output = NULL;
    return rc;
}
