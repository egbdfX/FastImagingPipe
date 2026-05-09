#include "SVDPreprocessGpu.h"

#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fitsio.h"

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/BasicSL/Complex.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableRecord.h>

namespace {

constexpr const char* kDefaultMsFiles[] = {"PSR0.ms", "PSR1.ms", "PSR2.ms"};

std::size_t flattened_index(std::size_t row, std::size_t chan, std::size_t num_rows) {
    return row + chan * num_rows;
}

struct RowAxisLayout {
    int channel_axis = 0;
    int pol_axis = 1;
    std::size_t num_channels = 0;
    std::size_t num_pols = 0;
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

void append_outputs(const HostPreprocessOutputs& src, HostPreprocessOutputs& dst) {
    dst.num_samples += src.num_samples;
    dst.bin.insert(dst.bin.end(), src.bin.begin(), src.bin.end());
    dst.vin.insert(dst.vin.end(), src.vin.begin(), src.vin.end());
    dst.vis_real.insert(dst.vis_real.end(), src.vis_real.begin(), src.vis_real.end());
    dst.vis_imag.insert(dst.vis_imag.end(), src.vis_imag.begin(), src.vis_imag.end());
}

std::vector<std::string> parse_ms_files(int argc, char** argv, std::string& output_prefix) {
    std::vector<std::string> ms_files;
    output_prefix.clear();

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--output-prefix") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --output-prefix");
            }
            output_prefix = argv[++i];
            continue;
        }
        ms_files.push_back(arg);
    }

    if (ms_files.empty()) {
        ms_files.assign(std::begin(kDefaultMsFiles), std::end(kDefaultMsFiles));
    }
    return ms_files;
}

}  // namespace

HostMeasurementSetData read_measurement_set(const std::string& ms_path) {
    using casacore::Array;
    using casacore::Complex;
    using casacore::ROArrayColumn;
    using casacore::Table;
    using casacore::Vector;

    Table vis(ms_path, Table::Old);
    const std::size_t num_rows = vis.nrow();

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

    for (std::size_t row = 0; row < result.num_rows; ++row) {
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

        result.uvw[row * 3 + 0] = static_cast<float>(uvw_row[0]);
        result.uvw[row * 3 + 1] = static_cast<float>(uvw_row[1]);
        result.uvw[row * 3 + 2] = static_cast<float>(uvw_row[2]);

        const std::size_t vis_channels = row_layout.num_channels;
        if (vis_channels == 0) {
            throw std::runtime_error("DATA has zero channels in " + ms_path);
        }

        for (std::size_t chan = 0; chan < result.num_channels; ++chan) {
            const std::size_t dst_idx = flattened_index(row, chan, result.num_rows);
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

int main(int argc, char** argv) {
    try {
        std::string output_prefix;
        const std::vector<std::string> ms_files = parse_ms_files(argc, argv, output_prefix);

        HostPreprocessOutputs aggregate;

        for (const auto& ms_file : ms_files) {
            std::cerr << "Processing " << ms_file << '\n';
            const HostMeasurementSetData host_data = read_measurement_set(ms_file);

            DevicePreprocessBuffers device_buffers;
            preprocess_measurement_set_gpu(host_data, device_buffers);

            HostPreprocessOutputs host_outputs;
            download_preprocess_outputs(device_buffers, host_outputs);
            free_device_preprocess_buffers(device_buffers);

            append_outputs(host_outputs, aggregate);
        }

        write_fits_2d(output_prefix + "Bin.fits", aggregate.bin, 2, static_cast<long>(aggregate.num_samples));
        write_fits_2d(output_prefix + "Vin.fits", aggregate.vin, 3, static_cast<long>(aggregate.vin.size() / 3));
        write_fits_2d(output_prefix + "Visreal.fits", aggregate.vis_real, 1, static_cast<long>(aggregate.num_samples));
        write_fits_2d(output_prefix + "Visimag.fits", aggregate.vis_imag, 1, static_cast<long>(aggregate.num_samples));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Preprocessing failed: " << ex.what() << '\n';
        return 1;
    }
}
