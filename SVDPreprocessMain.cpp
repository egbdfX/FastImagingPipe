#include "SVDPreprocessGpu.h"

#include <algorithm>
#include <complex>
#include <deque>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fitsio.h"

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/BasicSL/Complex.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableRecord.h>

namespace {

constexpr const char* kDefaultMsFiles[] = {"PSR0.ms", "PSR1.ms", "PSR2.ms"};

std::mutex g_casacore_mutex;
std::mutex g_log_mutex;

std::size_t flattened_index(std::size_t row, std::size_t chan, std::size_t num_rows) {
    return row + chan * num_rows;
}

struct RowAxisLayout {
    int channel_axis = 0;
    int pol_axis = 1;
    std::size_t num_channels = 0;
    std::size_t num_pols = 0;
};

struct PreprocessOptions {
    std::vector<std::string> ms_files;
    std::string output_prefix;
    std::string combined_output;
    bool group_by_time = false;
    std::size_t snapshot_workers = 0;
};

struct SnapshotWork {
    std::size_t index = 0;
    std::string ms_file;
    std::vector<std::size_t> selected_rows;
    bool use_selected_rows = false;
    double time_value = 0.0;
};

struct SnapshotResult {
    std::size_t index = 0;
    std::size_t num_samples = 0;
    HostPreprocessOutputs outputs;
};

RowAxisLayout detect_row_axis_layout(const casacore::IPosition& shape) {
    if (shape.nelements() != 2) {
        throw std::runtime_error("Expected a 2D row shape");
    }

    RowAxisLayout layout;
    const std::size_t dim0 = static_cast<std::size_t>(shape[0]);
    const std::size_t dim1 = static_cast<std::size_t>(shape[1]);

    if (dim1 == 4 || dim1 == 2 || dim1 == 1) {
        layout.channel_axis = 0;
        layout.pol_axis = 1;
        layout.num_channels = dim0;
        layout.num_pols = dim1;
        return layout;
    }
    if (dim0 == 4 || dim0 == 2 || dim0 == 1) {
        layout.channel_axis = 1;
        layout.pol_axis = 0;
        layout.num_channels = dim1;
        layout.num_pols = dim0;
        return layout;
    }

    if (dim0 >= dim1) {
        layout.channel_axis = 0;
        layout.pol_axis = 1;
        layout.num_channels = dim0;
        layout.num_pols = dim1;
    } else {
        layout.channel_axis = 1;
        layout.pol_axis = 0;
        layout.num_channels = dim1;
        layout.num_pols = dim0;
    }
    return layout;
}

casacore::IPosition make_row_index(const RowAxisLayout& layout, std::size_t chan, std::size_t pol) {
    casacore::IPosition index(2, 0, 0);
    index[layout.channel_axis] = static_cast<int>(chan);
    index[layout.pol_axis] = static_cast<int>(pol);
    return index;
}

void check_fits_status(int status, const std::string& context) {
    if (status != 0) {
        fits_report_error(stderr, status);
        throw std::runtime_error(context);
    }
}

void write_fits_2d(
    const std::string& filename,
    const std::vector<float>& values,
    long axis0,
    long axis1
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    long naxes[2] = {axis0, axis1};
    const std::string output_name = "!" + filename;

    fits_create_file(&fptr, output_name.c_str(), &status);
    check_fits_status(status, "Failed to create FITS file: " + filename);

    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    check_fits_status(status, "Failed to create FITS image: " + filename);

    const long total = axis0 * axis1;
    fits_write_img(
        fptr,
        TFLOAT,
        1,
        total,
        const_cast<float*>(values.data()),
        &status
    );
    check_fits_status(status, "Failed to write FITS image: " + filename);

    fits_close_file(fptr, &status);
    check_fits_status(status, "Failed to close FITS file: " + filename);
}

void write_fits_image_hdu(
    fitsfile* fptr,
    const std::string& filename,
    const char* extname,
    const std::vector<float>& values,
    int naxis,
    long* naxes
) {
    int status = 0;
    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
    check_fits_status(status, std::string("Failed to create FITS extension: ") + extname);

    fits_update_key(fptr, TSTRING, "EXTNAME", const_cast<char*>(extname), nullptr, &status);
    check_fits_status(status, std::string("Failed to name FITS extension: ") + extname);

    long total = 1;
    for (int axis = 0; axis < naxis; ++axis) {
        total *= naxes[axis];
    }

    if (static_cast<std::size_t>(total) != values.size()) {
        throw std::runtime_error("Internal FITS shape mismatch while writing " + filename);
    }

    fits_write_img(fptr, TFLOAT, 1, total, const_cast<float*>(values.data()), &status);
    check_fits_status(status, std::string("Failed to write FITS extension: ") + extname);
}

void write_combined_input_fits(
    const std::string& filename,
    const HostPreprocessOutputs& aggregate,
    std::size_t num_snapshots,
    std::size_t samples_per_snapshot
) {
    if (num_snapshots == 0 || samples_per_snapshot == 0) {
        throw std::runtime_error("Cannot write combined FITS with no snapshots");
    }
    if (aggregate.num_samples != num_snapshots * samples_per_snapshot) {
        throw std::runtime_error("Combined FITS dimensions do not match aggregate sample count");
    }
    if (aggregate.vin.size() != num_snapshots * 9) {
        throw std::runtime_error("Combined FITS dimensions do not match transformation matrices");
    }
    if (aggregate.bin.size() != aggregate.num_samples * 2) {
        throw std::runtime_error("Combined FITS dimensions do not match R-coordinate data");
    }
    if (aggregate.vis_real.size() != aggregate.num_samples ||
        aggregate.vis_imag.size() != aggregate.num_samples) {
        throw std::runtime_error("Combined FITS dimensions do not match visibility data");
    }

    std::vector<float> visibilities(num_snapshots * samples_per_snapshot * 2);
    for (std::size_t snapshot = 0; snapshot < num_snapshots; ++snapshot) {
        for (std::size_t sample = 0; sample < samples_per_snapshot; ++sample) {
            const std::size_t aggregate_idx = snapshot * samples_per_snapshot + sample;
            const std::size_t combined_idx = (snapshot * samples_per_snapshot + sample) * 2;
            visibilities[combined_idx + 0] = aggregate.vis_real[aggregate_idx];
            visibilities[combined_idx + 1] = aggregate.vis_imag[aggregate_idx];
        }
    }

    fitsfile* fptr = nullptr;
    int status = 0;
    const std::string output_name = "!" + filename;

    fits_create_file(&fptr, output_name.c_str(), &status);
    check_fits_status(status, "Failed to create FITS file: " + filename);

    fits_create_img(fptr, BYTE_IMG, 0, nullptr, &status);
    check_fits_status(status, "Failed to create primary FITS HDU: " + filename);

    long v_axes[3] = {
        3,
        3,
        static_cast<long>(num_snapshots)
    };
    write_fits_image_hdu(
        fptr,
        filename,
        "TRANSFORMATION MATRICES",
        aggregate.vin,
        3,
        v_axes
    );

    long sample_axes[3] = {
        2,
        static_cast<long>(samples_per_snapshot),
        static_cast<long>(num_snapshots)
    };
    write_fits_image_hdu(
        fptr,
        filename,
        "VISIBILITIES",
        visibilities,
        3,
        sample_axes
    );
    write_fits_image_hdu(
        fptr,
        filename,
        "R-COORDINATES",
        aggregate.bin,
        3,
        sample_axes
    );

    fits_close_file(fptr, &status);
    check_fits_status(status, "Failed to close FITS file: " + filename);
}

void append_padded_outputs(
    const HostPreprocessOutputs& src,
    std::size_t padded_samples_per_snapshot,
    HostPreprocessOutputs& dst
) {
    if (src.num_samples > padded_samples_per_snapshot) {
        throw std::runtime_error("Snapshot has more samples than the padded snapshot size");
    }
    if (src.bin.size() != src.num_samples * 2 ||
        src.vis_real.size() != src.num_samples ||
        src.vis_imag.size() != src.num_samples ||
        src.vin.size() != 9) {
        throw std::runtime_error("Snapshot output dimensions are inconsistent");
    }

    dst.num_samples += padded_samples_per_snapshot;

    dst.bin.insert(dst.bin.end(), src.bin.begin(), src.bin.end());
    dst.bin.insert(dst.bin.end(), (padded_samples_per_snapshot - src.num_samples) * 2, 0.0f);

    dst.vin.insert(dst.vin.end(), src.vin.begin(), src.vin.end());

    dst.vis_real.insert(dst.vis_real.end(), src.vis_real.begin(), src.vis_real.end());
    dst.vis_real.insert(dst.vis_real.end(), padded_samples_per_snapshot - src.num_samples, 0.0f);

    dst.vis_imag.insert(dst.vis_imag.end(), src.vis_imag.begin(), src.vis_imag.end());
    dst.vis_imag.insert(dst.vis_imag.end(), padded_samples_per_snapshot - src.num_samples, 0.0f);
}

PreprocessOptions parse_options(int argc, char** argv) {
    PreprocessOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--output-prefix") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --output-prefix");
            }
            options.output_prefix = argv[++i];
            continue;
        }
        if (arg == "--combined-output") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --combined-output");
            }
            options.combined_output = argv[++i];
            continue;
        }
        if (arg == "--group-by-time") {
            options.group_by_time = true;
            continue;
        }
        if (arg == "--snapshot-workers") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --snapshot-workers");
            }
            options.snapshot_workers = static_cast<std::size_t>(std::stoul(argv[++i]));
            if (options.snapshot_workers == 0) {
                throw std::runtime_error("--snapshot-workers must be greater than zero");
            }
            continue;
        }
        options.ms_files.push_back(arg);
    }

    if (options.ms_files.empty()) {
        options.ms_files.assign(std::begin(kDefaultMsFiles), std::end(kDefaultMsFiles));
    }
    if (options.combined_output.empty()) {
        options.combined_output = options.output_prefix + "input.fits";
    }
    return options;
}

std::size_t resolve_snapshot_workers(const PreprocessOptions& options, std::size_t num_snapshots) {
    if (num_snapshots == 0) {
        return 1;
    }
    if (options.snapshot_workers != 0) {
        return std::min(options.snapshot_workers, num_snapshots);
    }

    const unsigned int hardware_workers = std::thread::hardware_concurrency();
    const std::size_t auto_workers =
        hardware_workers == 0 ? 2 : static_cast<std::size_t>(hardware_workers);
    return std::max<std::size_t>(1, std::min<std::size_t>(4, std::min(auto_workers, num_snapshots)));
}

SnapshotResult process_snapshot(const SnapshotWork& work) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        std::cerr << "Processing snapshot " << (work.index + 1) << ": " << work.ms_file;
        if (work.use_selected_rows) {
            std::cerr << " TIME " << work.time_value
                      << " rows " << work.selected_rows.size();
        }
        std::cerr << '\n';
    }

    HostMeasurementSetData host_data;
    {
        std::lock_guard<std::mutex> lock(g_casacore_mutex);
        host_data = work.use_selected_rows
            ? read_measurement_set_rows(work.ms_file, work.selected_rows)
            : read_measurement_set(work.ms_file);
    }

    DevicePreprocessBuffers device_buffers;
    preprocess_measurement_set_gpu(host_data, device_buffers);

    SnapshotResult result;
    try {
        result.index = work.index;
        result.num_samples = host_data.num_samples;
        download_preprocess_outputs(device_buffers, result.outputs);
        free_device_preprocess_buffers(device_buffers);
    } catch (...) {
        free_device_preprocess_buffers(device_buffers);
        throw;
    }
    return result;
}

std::vector<std::pair<double, std::vector<std::size_t>>> group_rows_by_time(
    const std::string& ms_path
) {
    using casacore::ROScalarColumn;
    using casacore::Table;

    Table vis(ms_path, Table::Old);
    ROScalarColumn<double> time_col(vis, "TIME");

    std::map<double, std::vector<std::size_t>> grouped_rows;
    for (std::size_t row = 0; row < vis.nrow(); ++row) {
        const double time_value = time_col(static_cast<casacore::rownr_t>(row));
        grouped_rows[time_value].push_back(row);
    }

    std::vector<std::pair<double, std::vector<std::size_t>>> time_groups;
    time_groups.reserve(grouped_rows.size());
    for (const auto& group : grouped_rows) {
        time_groups.push_back(std::make_pair(group.first, group.second));
    }
    return time_groups;
}

}  // namespace

HostMeasurementSetData read_measurement_set_rows(
    const std::string& ms_path,
    const std::vector<std::size_t>& selected_rows
) {
    using casacore::Array;
    using casacore::Complex;
    using casacore::ROArrayColumn;
    using casacore::Table;
    using casacore::Vector;

    Table vis(ms_path, Table::Old);
    const std::size_t table_rows = vis.nrow();
    const std::size_t num_rows = selected_rows.size();

    if (num_rows == 0) {
        throw std::runtime_error("No rows selected from " + ms_path);
    }

    ROArrayColumn<double> uvw_col(vis, "UVW");
    ROArrayColumn<Complex> data_col(vis, "DATA");
    ROArrayColumn<bool> flag_col(vis, "FLAG");

    const bool has_weight_spectrum = vis.tableDesc().isColumn("WEIGHT_SPECTRUM");
    std::unique_ptr<ROArrayColumn<float>> weight_col;
    if (has_weight_spectrum) {
        weight_col.reset(new ROArrayColumn<float>(vis, "WEIGHT_SPECTRUM"));
    }

    Table spw = vis.keywordSet().asTable("SPECTRAL_WINDOW");
    ROArrayColumn<double> chan_freq_col(spw, "CHAN_FREQ");

    std::vector<float> frequencies_hz;
    for (std::size_t row = 0; row < spw.nrow(); ++row) {
        Vector<double> row_freq;
        chan_freq_col.get(static_cast<casacore::rownr_t>(row), row_freq);
        for (casacore::uInt chan = 0; chan < row_freq.size(); ++chan) {
            frequencies_hz.push_back(static_cast<float>(row_freq[chan]));
        }
    }

    if (frequencies_hz.empty()) {
        throw std::runtime_error("No frequencies found in SPECTRAL_WINDOW for " + ms_path);
    }

    HostMeasurementSetData result;
    result.num_rows = num_rows;
    result.num_channels = frequencies_hz.size();
    result.num_samples = result.num_rows * result.num_channels;
    result.frequencies_hz = std::move(frequencies_hz);
    result.uvw.resize(result.num_rows * 3);
    result.vis0_real.resize(result.num_samples);
    result.vis0_imag.resize(result.num_samples);
    result.vis3_real.resize(result.num_samples);
    result.vis3_imag.resize(result.num_samples);
    result.flag0.resize(result.num_samples);
    result.flag3.resize(result.num_samples);
    result.weight0.resize(result.num_samples, 1.0f);
    result.weight3.resize(result.num_samples, 1.0f);

    for (std::size_t local_row = 0; local_row < result.num_rows; ++local_row) {
        const std::size_t row = selected_rows[local_row];
        if (row >= table_rows) {
            throw std::runtime_error("Selected row is outside " + ms_path);
        }

        Vector<double> uvw_row;
        Array<Complex> data_row;
        Array<bool> flag_row;

        uvw_col.get(static_cast<casacore::rownr_t>(row), uvw_row);
        data_col.get(static_cast<casacore::rownr_t>(row), data_row);
        flag_col.get(static_cast<casacore::rownr_t>(row), flag_row);

        if (uvw_row.size() != 3) {
            throw std::runtime_error("UVW does not have 3 entries in " + ms_path);
        }

        const auto data_shape = data_row.shape();
        if (data_shape.nelements() != 2) {
            throw std::runtime_error("DATA does not have 2 dimensions in " + ms_path);
        }

        const RowAxisLayout row_layout = detect_row_axis_layout(data_shape);
        if (row_layout.num_pols < 4) {
            throw std::runtime_error("DATA has fewer than 4 polarisations in " + ms_path);
        }

        Array<float> weight_row;
        if (has_weight_spectrum) {
            weight_col->get(static_cast<casacore::rownr_t>(row), weight_row);
        }

        result.uvw[local_row * 3 + 0] = static_cast<float>(uvw_row[0]);
        result.uvw[local_row * 3 + 1] = static_cast<float>(uvw_row[1]);
        result.uvw[local_row * 3 + 2] = static_cast<float>(uvw_row[2]);

        const std::size_t vis_channels = row_layout.num_channels;
        if (vis_channels == 0) {
            throw std::runtime_error("DATA has zero channels in " + ms_path);
        }

        for (std::size_t chan = 0; chan < result.num_channels; ++chan) {
            const std::size_t dst_idx = flattened_index(local_row, chan, result.num_rows);
            const std::size_t chan_in_row = chan % vis_channels;
            const casacore::IPosition pol0_index = make_row_index(row_layout, chan_in_row, 0);
            const casacore::IPosition pol3_index = make_row_index(row_layout, chan_in_row, 3);
            const Complex vis0_value = data_row(pol0_index);
            const Complex vis3_value = data_row(pol3_index);

            result.vis0_real[dst_idx] = vis0_value.real();
            result.vis0_imag[dst_idx] = vis0_value.imag();
            result.vis3_real[dst_idx] = vis3_value.real();
            result.vis3_imag[dst_idx] = vis3_value.imag();
            result.flag0[dst_idx] = flag_row(pol0_index) ? 1U : 0U;
            result.flag3[dst_idx] = flag_row(pol3_index) ? 1U : 0U;

            if (has_weight_spectrum) {
                result.weight0[dst_idx] = weight_row(pol0_index);
                result.weight3[dst_idx] = weight_row(pol3_index);
            }
        }
    }

    return result;
}

HostMeasurementSetData read_measurement_set(const std::string& ms_path) {
    using casacore::Table;

    Table vis(ms_path, Table::Old);
    std::vector<std::size_t> all_rows(vis.nrow());
    for (std::size_t row = 0; row < all_rows.size(); ++row) {
        all_rows[row] = row;
    }
    return read_measurement_set_rows(ms_path, all_rows);
}

int main(int argc, char** argv) {
    try {
        const PreprocessOptions options = parse_options(argc, argv);

        std::vector<SnapshotWork> snapshots;

        for (const auto& ms_file : options.ms_files) {
            if (options.group_by_time) {
                const auto time_groups = group_rows_by_time(ms_file);
                std::cerr << "Processing " << ms_file << " as "
                          << time_groups.size() << " TIME snapshots\n";

                for (const auto& time_group : time_groups) {
                    SnapshotWork work;
                    work.index = snapshots.size();
                    work.ms_file = ms_file;
                    work.selected_rows = time_group.second;
                    work.use_selected_rows = true;
                    work.time_value = time_group.first;
                    snapshots.push_back(std::move(work));
                }
                continue;
            }

            SnapshotWork work;
            work.index = snapshots.size();
            work.ms_file = ms_file;
            snapshots.push_back(std::move(work));
        }

        const std::size_t num_snapshots = snapshots.size();
        const std::size_t snapshot_workers = resolve_snapshot_workers(options, num_snapshots);
        std::cerr << "Running " << num_snapshots << " snapshot(s) with "
                  << snapshot_workers << " worker(s)\n";

        std::vector<SnapshotResult> results(num_snapshots);
        std::deque<std::future<SnapshotResult>> running;
        std::size_t next_snapshot = 0;

        while (next_snapshot < num_snapshots || !running.empty()) {
            while (next_snapshot < num_snapshots && running.size() < snapshot_workers) {
                const SnapshotWork work = snapshots[next_snapshot++];
                running.push_back(std::async(std::launch::async, [work]() {
                    return process_snapshot(work);
                }));
            }

            SnapshotResult result = running.front().get();
            running.pop_front();
            results[result.index] = std::move(result);
        }

        HostPreprocessOutputs aggregate;
        std::size_t samples_per_snapshot = 0;
        for (const SnapshotResult& result : results) {
            samples_per_snapshot = std::max(samples_per_snapshot, result.num_samples);
        }

        std::size_t padded_snapshots = 0;
        for (const SnapshotResult& result : results) {
            if (result.num_samples != samples_per_snapshot) {
                ++padded_snapshots;
            }
            append_padded_outputs(result.outputs, samples_per_snapshot, aggregate);
        }

        write_fits_2d(options.output_prefix + "Bin.fits", aggregate.bin, 2, static_cast<long>(aggregate.num_samples));
        write_fits_2d(options.output_prefix + "Vin.fits", aggregate.vin, 3, static_cast<long>(aggregate.vin.size() / 3));
        write_fits_2d(options.output_prefix + "Visreal.fits", aggregate.vis_real, 1, static_cast<long>(aggregate.num_samples));
        write_fits_2d(options.output_prefix + "Visimag.fits", aggregate.vis_imag, 1, static_cast<long>(aggregate.num_samples));
        write_combined_input_fits(
            options.combined_output,
            aggregate,
            num_snapshots,
            samples_per_snapshot
        );
        std::cerr << "Wrote " << num_snapshots << " snapshot(s), "
                  << samples_per_snapshot << " padded samples per snapshot\n";
        if (padded_snapshots != 0) {
            std::cerr << "Zero-padded " << padded_snapshots
                      << " snapshot(s) to match the maximum sample count\n";
        }
        std::cerr
                  << "Wrote combined FITS input: "
                  << options.combined_output << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Preprocessing failed: " << ex.what() << '\n';
        return 1;
    }
}
