#define main_pipe main_image_pipe_unused
#include "main_pipe.cpp"
#undef main_pipe

static void image_cb(void* userdata0,
                     void* userdata1,
                     void* image,
                     size_t image_size,
                     size_t iter){
    fip_pipe_iter_state* state = (fip_pipe_iter_state*)userdata0;
    (void)userdata1;
    (void)iter;

    long fpixel[2] = {1, 1};
    long lpixel[2] = {(long)image_size, (long)image_size};
    fits_write_subset(state->output, TFLOAT, fpixel, lpixel, image, &state->output_status);

    if(state->output_status){
        fits_report_error(stderr, state->output_status);
        fflush(stderr);
    }
}

extern "C" int main_image(int argc, char* argv[]){
    int rc = EXIT_FAILURE;
    int fits_status = 0;
    char* input_name = NULL;
    char* output_name = NULL;
    fitsfile* output = NULL;

    fip_pipe_cuda_state* pipe = NULL;
    fip_pipe_iter_state state;

    long long image_size = 1024;
    long long unit_size = 32;
    long long num_baselines = NUM_BASELINES_FLAG_DEFAULT;
    long long snapshot = 0;
    int verbose = 0;
    int gpu_ordinal = 0;
    float cell_size = 0.000020595;
    long long snap_count_file = 0;
    long long num_baselines_file = 0;
    size_t num_channels = 0;
    size_t num_rows_max = 0;
    size_t num_rows;
    size_t unit_num;
    Table input, spw;
    TableIterator iter;
    ROArrayColumn<double> chan_freq_col;

    struct poptOption opts[] = {
        {"help", 'h', POPT_ARG_NONE, NULL, HELP_FLAG, "Print this help", NULL},
        {"input", 'i', POPT_ARG_STRING, &input_name, 0, "Input MS", "FILE"},
        {"output", 'o', POPT_ARG_STRING, &output_name, 0, "Output FITS image", "FILE"},
        {"snapshot", 0, POPT_ARG_LONGLONG, &snapshot, 0, "Snapshot index", "N"},
        {"image-size", 's', POPT_ARG_LONGLONG, &image_size, 0, "Image Size", "N"},
        {"num-baselines", 'b', POPT_ARG_LONGLONG, &num_baselines, 0, "Number of baselines", "N"},
        {"cell-size", 'C', POPT_ARG_FLOAT, &cell_size, 0, "Cell Size", "F>0.0"},
        {"unit-size", 'u', POPT_ARG_LONGLONG, &unit_size, 0, "Unit Size", "N"},
        {"gpu", 'g', POPT_ARG_INT, &gpu_ordinal, 0, "GPU ordinal", "N"},
        {"verbose", 'v', POPT_ARG_NONE, NULL, VERBOSE_FLAG, "Increase verbosity", NULL},
        {"quiet", 'q', POPT_ARG_NONE, NULL, QUIET_FLAG, "Decrease verbosity", NULL},
        {NULL, 0, POPT_ARG_NONE, NULL, 0, NULL, NULL},
    };

    poptContext parser = poptGetContext("fip-image", argc, (const char**)argv, opts, 0);
    while((rc = poptGetNextOpt(parser)) > 0){
        if(rc == HELP_FLAG){
            poptPrintHelp(parser, stdout, 0);
            poptFreeContext(parser);
            return EXIT_SUCCESS;
        }else if(rc == VERBOSE_FLAG){
            verbose += 10;
        }else if(rc == QUIET_FLAG){
            verbose -= 10;
        }
    }
    poptFreeContext(parser);

    input_name = fip_strdup(input_name ? input_name : "input.ms");
    output_name = fip_strdup(output_name ? output_name : "image.fits");

    if(!Table::isReadable(input_name)){
        fprintf(stderr, "Error: %s is not readable!\n", input_name);
        goto finalexit;
    }

    input = Table(input_name, Table::Old);
    spw = input.keywordSet().asTable("SPECTRAL_WINDOW");
    iter = TableIterator(input, "TIME", TableIterator::Ascending, TableIterator::QuickSort);

    chan_freq_col = ROArrayColumn<double>(spw, "CHAN_FREQ");
    if(chan_freq_col.shapeColumn().empty()){
        for(size_t i = 0; i < chan_freq_col.nrow(); i++)
            num_channels += chan_freq_col.shape(i).product();
    }else{
        num_channels = (size_t)chan_freq_col.shapeColumn().product();
    }

    for(snap_count_file = 0; !iter.pastEnd(); iter++){
        num_rows = (size_t)iter.table().nrow();
        if(num_rows == 0) continue;
        snap_count_file++;
        num_rows_max = num_rows_max > num_rows ? num_rows_max : num_rows;
    }

    if(snapshot < 0 || snapshot >= snap_count_file){
        fprintf(stderr, "Error: --snapshot=%lld outside valid range [0,%lld)\n",
                snapshot, snap_count_file);
        goto finalexit;
    }

    num_baselines_file = num_rows_max * num_channels;
    if(num_baselines == NUM_BASELINES_FLAG_DEFAULT)
        num_baselines = num_baselines_file;
    unit_num = image_size / unit_size;

    {
        char diskfile[4096];
        snprintf(diskfile, sizeof(diskfile), "!%s", output_name);
        long long axes[2] = {image_size, image_size};
        fits_create_diskfile(&output, diskfile, &fits_status);
        fits_create_imgll(output, FLOAT_IMG, 2, axes, &fits_status);
        if(fits_status) goto fitsfail;
    }

    state = fip_pipe_iter_state(input, output).skip((size_t)snapshot);

    if(fip_pipe_cuda_alloc(&pipe, verbose, gpu_ordinal, num_baselines, image_size, cell_size, unit_size, unit_num))
        goto fitsfail;

    rc = fip_pipe_cuda(pipe, input_cb, NULL, image_cb, &state, NULL, (size_t)snapshot, (size_t)snapshot + 1, 1);

    fip_pipe_cuda_clear(&pipe);
    fits_flush_file(output, &fits_status);

fitsfail:
    if(fits_status)
        fits_report_error(stderr, fits_status);
    if(output)
        fits_close_file(output, &fits_status);

finalexit:
    free(input_name);
    free(output_name);
    return rc;
}
