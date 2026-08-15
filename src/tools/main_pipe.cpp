#include "main.h"

#include <stdint.h>
#include <string.h>
#include <fitsio.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/ArrayAccessor.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Quanta/MVTime.h>
#include <casacore/casa/Quanta/Unit.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableIter.h>
#include <casacore/tables/Tables/TableRecord.h>
#include <casacore/tables/TaQL.h>

#include "fip/cuda_pipeline.h"
#include "fits_utils.h"


#define  START_OFFSET_FLAG       4
#define  END_OFFSET_FLAG         5
#define  SNAP_COUNT_GE_3_FLAG    6
#define  NUM_BASELINES_FLAG      7

#define  NUM_BASELINES_FLAG_DEFAULT      -1
#define  START_OFFSET_FLAG_DEFAULT       -1
#define  END_OFFSET_FLAG_DEFAULT          0
#define  SNAP_COUNT_FLAG_DEFAULT         -1

#define  K_SPEED_OF_LIGHT         299792458.0f


/* We heavily use CASACore objects, so import the namespace. */
using namespace casacore;


/* fip pipe implementation. */
struct fip_pipe_iter_state{
    fip_pipe_iter_state(Table input=Table(), fitsfile* output=NULL) :
                        input(input),
                        input_iteration(0),
                        array_center(0),
                        output(output),
                        output_status(0){
        if(!input.isNull())
            reset_iterator();
    }

    fip_pipe_iter_state& reset_iterator(void){
        iter = TableIterator(input, "TIME", TableIterator::Ascending,
                                            TableIterator::QuickSort);
        input_iteration = 0;
        chan_freq       = ROArrayColumn<Double>(input.keywordSet()
                                                     .asTable("SPECTRAL_WINDOW"),
                                                "CHAN_FREQ").getColumn();

        Table observation = input.keywordSet().asTable("OBSERVATION");
        if(observation.nrow() > 0 &&
           observation.tableDesc().isColumn("ARRAY_CENTER")){
            ROArrayColumn<Double> array_center_col(observation, "ARRAY_CENTER");
            Vector<Double> array_center_vec = array_center_col.get(0);
            if(array_center_vec.size() > 0){
                array_center = array_center_vec[0];
            }
        }
        return *this;
    }

    fip_pipe_iter_state& skip(size_t num){
        for(size_t i=0;i<num;i++){
            input_iteration++;
            iter.next();
        }
        return *this;
    }

    void next(void*    userdata,
              Complex* visibilities,
              Float*   coordinates,
              Float    transform[3][3],
              size_t   num_baselines,
              size_t   iteration){
        (void)userdata;
        (void)iteration;


        /**
         * Get current snapshot. Obtain vital information about it.
         */

        Table  snapshot = iter.table(); iter.next();
        size_t num_rows = (size_t)snapshot.nrow();
        const bool has_weight_spectrum = snapshot.tableDesc().isColumn("WEIGHT_SPECTRUM");


        /**
         * Open snapshot's columns with accessors.
         *
         * Also, get vital shape information from the first row of "DATA".
         * MeasurementSet 2.0 specifies [1] that the shape of every row of "DATA" is
         *
         *     (DATA): Complex(Nc, Nf)
         *
         * where:
         *
         *     Nc = Number of correlators
         *     Nf = Number of frequency channels
         *
         * Here we take num_pols=Nc, and num_channels=Nf.
         *
         * [1] https://casacore.github.io/casacore-notes/229.html#x1-630005.1
         */

        ROArrayColumn<Double>  uvw_col (snapshot, "UVW"); /* Raw Coordinates */
        ROArrayColumn<Complex> data_col(snapshot, "DATA");/* Raw Visibilities */
        ROArrayColumn<Bool>    flag_col(snapshot, "FLAG");/* Flag vector */
        ROArrayColumn<Float>   weight_col;                /* Optional weights */
        if(has_weight_spectrum)
            weight_col = ROArrayColumn<Float>(snapshot, "WEIGHT_SPECTRUM");
        size_t num_pols       = data_col.shape(0)[0];
        size_t num_channels   = data_col.shape(0)[1];
        bool   merge_two_pols = num_pols > 1;
        if(num_baselines < num_rows*num_channels)
            throw std::runtime_error("DATA has more baselines than expected!");
        if(num_pols != 1 && num_pols != 2 && num_pols != 4)
            throw std::runtime_error("DATA has unexpected number of "
                                     "polarizations, neither 1 nor 2 nor 4!");


        /**
         * To optimize data-loading, use a Slicer object designed to select the
         * one or two polarizations only we want: Either {0}, {0,1} or {0,3}.
         *
         * We use this slicer in the identically-shaped arrays "DATA" and "FLAG".
         */

        Slice  pol_slice = !merge_two_pols ? Slice(0) : Slice(0, 2, num_pols-1);
        Slicer pol_slicer{pol_slice, Slice()};
        Array<Double>  uvw  = uvw_col   .getColumn();
        Array<Complex> data = data_col  .getColumn(pol_slicer);
        Array<Bool>    flag = flag_col  .getColumn(pol_slicer);
        Array<Float>   weight;
        if(has_weight_spectrum){
            weight          = weight_col.getColumn(pol_slicer);
        }


        /**
         * We now have the data. Process it. Our loops assume contiguous storage.
         *
         * Fill the visibilities array under control of the flags and weights.
         * Then, compute the coordinates.
         */

        if(!uvw   .contiguousStorage())
            throw std::runtime_error("Storage unexpectedly not contiguous!");
        if( uvw   .steps()[0] != 1)
            throw std::runtime_error("Unexpected stride!");
        if(!data  .contiguousStorage())
            throw std::runtime_error("Storage unexpectedly not contiguous!");
        if( data  .steps()[0] != 1)
            throw std::runtime_error("Unexpected stride!");
        if(!flag  .contiguousStorage())
            throw std::runtime_error("Storage unexpectedly not contiguous!");
        if( flag  .steps()[0] != 1)
            throw std::runtime_error("Unexpected stride!");
        if(!weight.contiguousStorage())
            throw std::runtime_error("Storage unexpectedly not contiguous!");
        if( weight.steps()[0] != 1)
            throw std::runtime_error("Unexpected stride!");

        Double*  ptr_uvw    = uvw .data();
        Complex* ptr_data   = data.data();
        Bool*    ptr_flag   = flag.data();
        Float*   ptr_weight = weight.empty() ? NULL : weight.data();
        Double*  ptr_freq   = chan_freq.data();


        /* Visibilities */
        for(size_t i=0;i<num_rows;i++){
            for(size_t j=0;j<num_channels;j++){
                Complex vis0    = *ptr_data++;
                Complex vis3    = merge_two_pols ? *ptr_data++ : 0;
                Bool    flag0   = *ptr_flag++;
                Bool    flag3   = merge_two_pols ? *ptr_flag++ : 1;
                Float   weight0 = has_weight_spectrum                   ? *ptr_weight++ : 1.0f;
                Float   weight3 = merge_two_pols ? (has_weight_spectrum ? *ptr_weight++ : 1.0f) : 0.0f;

                Float   w0      = flag0 ? 0.0f : weight0;
                Float   w3      = flag3 ? 0.0f : weight3;
                Float   w_sum   = weight0+weight3;
                if(w_sum > 0.0f){
                    *visibilities++ = (vis0*w0 + vis3*w3)/w_sum;
                }else{
                    *visibilities++ = 0;
                }
            }
        }


        /* Coordinates, Part I: Mean. */
        Double x0avg=0, x1avg=0, x2avg=0, x0, x1, x2, u, v, w;
        for(size_t i=0;i<num_rows;i++){
            u = ptr_uvw[3*i+0];
            v = ptr_uvw[3*i+1];
            w = ptr_uvw[3*i+2];

            for(size_t j=0;j<num_channels;j++){
                x0 = u * (ptr_freq[j] / K_SPEED_OF_LIGHT);
                x1 = v * (ptr_freq[j] / K_SPEED_OF_LIGHT);
                x2 = w * (ptr_freq[j] / K_SPEED_OF_LIGHT);

                x0avg += x0;
                x1avg += x1;
                x2avg += x2;
            }
        }
        x0avg /= num_baselines;
        x1avg /= num_baselines;
        x2avg /= num_baselines;


        /* Coordinates, Part II: Covariance. */
        Double covariance[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};
        for(size_t i=0;i<num_rows;i++){
            u = ptr_uvw[3*i+0];
            v = ptr_uvw[3*i+1];
            w = ptr_uvw[3*i+2];

            for(size_t j=0;j<num_channels;j++){
                x0 = u * (ptr_freq[j] / K_SPEED_OF_LIGHT) - x0avg;
                x1 = v * (ptr_freq[j] / K_SPEED_OF_LIGHT) - x1avg;
                x2 = w * (ptr_freq[j] / K_SPEED_OF_LIGHT) - x2avg;

                covariance[0][0] += x0*x0;
                covariance[0][1] += x0*x1;
                covariance[0][2] += x0*x2;
                covariance[1][1] += x1*x1;
                covariance[1][2] += x1*x2;
                covariance[2][2] += x2*x2;
            }
        }
        covariance[0][0] /= num_baselines;
        covariance[0][1] /= num_baselines;
        covariance[0][2] /= num_baselines;
        covariance[1][1] /= num_baselines;
        covariance[1][2] /= num_baselines;
        covariance[2][2] /= num_baselines;
        covariance[1][0]  = covariance[0][1];
        covariance[2][0]  = covariance[0][2];
        covariance[2][1]  = covariance[1][2];


        /* Coordinates, Part III: SVD. */
        Double eigenvalues    [3] =  {0,0,0};
        Double eigenvectors[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}};
        /* FILL ME! */
        for(size_t i=0;i<3;i++)
            for(size_t j=0;j<3;j++)
                transform[i][j] = eigenvectors[i][j];


        /* Coordinates, Part IV: Project. */
        for(size_t i=0;i<num_rows;i++){
            u = ptr_uvw[3*i+0];
            v = ptr_uvw[3*i+1];
            w = ptr_uvw[3*i+2];

            for(size_t j=0;j<num_channels;j++){
                x0 = u * (ptr_freq[j] / K_SPEED_OF_LIGHT) - x0avg;
                x1 = v * (ptr_freq[j] / K_SPEED_OF_LIGHT) - x1avg;
                x2 = w * (ptr_freq[j] / K_SPEED_OF_LIGHT) - x2avg;

                coordinates[2*(num_channels*i+j) + 0] = eigenvectors[0][0]*x0 +
                                                        eigenvectors[0][1]*x1 +
                                                        eigenvectors[0][2]*x2;
                coordinates[2*(num_channels*i+j) + 1] = eigenvectors[1][0]*x0 +
                                                        eigenvectors[1][1]*x1 +
                                                        eigenvectors[1][2]*x2;
            }
        }
    }

    Table         input;
    size_t        input_iteration;
    TableIterator iter;
    Array<Double> chan_freq;
    double        array_center;
    fitsfile*     output;
    int           output_status;
};

static char* fip_strdup(const char* s){
    char*  t;
    size_t l;

    if(!s)
        return NULL;

    l = strlen(s);
    t = (char*)malloc(l+1);
    if(!t)
        return NULL;

    return (char*)memcpy(t, s, l+1);
}

static void input_cb (void*  userdata0,
                      void*  userdata1,
                      void*  visibilities,
                      float* coordinates,
                      float  transform[3][3],
                      size_t num_baselines,
                      size_t iteration){
    ((fip_pipe_iter_state*)userdata0)->next(userdata1,
                                            (Complex*)visibilities,
                                            coordinates,
                                            transform,
                                            num_baselines,
                                            iteration);
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

    long fres[3] = {1, 1,                           (long)iter+1-2};
    long lres[3] = {(long)unit_num, (long)unit_num, (long)iter+1-2};

    fits_write_subset(state->output, TFLOAT, fres, lres, result, &state->output_status);

    if(state->output_status){
        fits_report_error(stderr, state->output_status);
        fflush(stderr);
    }
}


int main_pipe(int argc, char* argv[]){
    int       rc              = EXIT_FAILURE;
    int       fits_status     = 0;

    char*     input_name      = NULL;
    fitsfile* output          = NULL;
    char*     output_name     = NULL;
    const char** leftovers    = NULL;

    fip_pipe_cuda_state* pipe = NULL;
    fip_pipe_iter_state  state;


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
    size_t    num_channels    = 0;
    size_t    num_rows_max    = 0;
    size_t    num_rows;
    size_t    unit_num;
    size_t    snap_count_file_out;
    size_t    i;


    Table                 input, spw, subset;
    TableIterator         iter;
    ROArrayColumn<double> chan_freq_col;


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
    input_name  = fip_strdup(input_name);
    output_name = fip_strdup(output_name);

    /**
     * Interpret first two leftover arguments as input and output file names,
     * if either of these remain unset.
     * 
     * If still not set, choose defaults "input.fits" and "output.fits".
     */

    leftovers = poptGetArgs(parser);
    if(leftovers){
        if(leftovers[0]){
            if(!input_name){
                input_name = fip_strdup(leftovers[0]);
                if(leftovers[1] && !output_name)
                    output_name = fip_strdup(leftovers[1]);
            }else if(!output_name)
                output_name = fip_strdup(leftovers[0]);
        }
    }
    if(!input_name)
        input_name  = fip_strdup("input.fits");
    if(!output_name)
        output_name = fip_strdup("output.fits");

    poptfail:
    poptFreeContext(parser);
    if(rc != EXIT_SUCCESS)
        goto finalexit;
    rc = EXIT_FAILURE;


    /**
     * Open input file.
     *
     * Also collect the file's vital statistics, enabling its validation and
     * that of the program's other arguments.
     */

    if(!Table::isReadable(input_name)){
        fprintf(stderr, "Error: %s is not readable!\n", input_name);
        goto fitsfail;
    }
    input  = Table(input_name, Table::Old);
    spw    = input.keywordSet().asTable("SPECTRAL_WINDOW");
    subset = input;
    iter   = TableIterator(subset, "TIME", TableIterator::Ascending,
                                           TableIterator::QuickSort);


    chan_freq_col = ROArrayColumn<double>(spw, "CHAN_FREQ");
    if(chan_freq_col.shapeColumn().empty())
        for(i=0; i<chan_freq_col.nrow(); i++)
            num_channels += chan_freq_col.shape(i).product();
    else
        num_channels = (size_t)chan_freq_col.shapeColumn().product();


    for(snap_count_file=0; !iter.pastEnd(); iter++){
        num_rows = (size_t)iter.table().nrow();
        if(num_rows == 0)
            continue;

        snap_count_file++;
        num_rows_max = num_rows_max > num_rows ?
                       num_rows_max : num_rows;
    }
    num_baselines_file = num_rows_max * num_channels;


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

    unit_num            = image_size/unit_size;
    snap_count_file_out = snap_count_file-2;
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
    if(fits_status)
        goto fitsfail;


    /* Execute Pipeline */
    state = fip_pipe_iter_state(subset, output).skip(snap_start_final);
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
    if(output)
        fits_close_file(output, &fits_status), output = NULL;

    finalexit:
    free(input_name);
    free(output_name);
    return rc;
}
