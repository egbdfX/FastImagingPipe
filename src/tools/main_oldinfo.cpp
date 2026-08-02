#include "main.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/BasicSL/Complex.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableRecord.h>

struct HostMeasurementSetData {
    std::size_t num_rows = 0;
    std::size_t num_channels = 0;
    std::size_t num_samples = 0;
    std::vector<float> uvw;
    std::vector<float> frequencies_hz;
    std::vector<float> vis0_real;
    std::vector<float> vis0_imag;
    std::vector<float> vis3_real;
    std::vector<float> vis3_imag;
    std::vector<std::uint8_t> flag0;
    std::vector<std::uint8_t> flag3;
    std::vector<float> weight0;
    std::vector<float> weight3;
};

struct MeasurementSetSnapshot {
    std::size_t index = 0;
    double time = 0.0;
    std::vector<std::size_t> rows;
    HostMeasurementSetData data;
};

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

    for (const auto& group : grouped_rows) {
        time_groups.push_back(std::make_pair(group.first, group.second));
    }

    return time_groups;
}

std::vector<MeasurementSetSnapshot> read_measurement_set_snapshots(
    const std::string& ms_path
) {
    std::vector<MeasurementSetSnapshot> snapshots;

    const auto time_groups = group_rows_by_time(ms_path);

    for (const auto& time_group : time_groups) {
        MeasurementSetSnapshot snapshot;

        snapshot.index = snapshots.size();
        snapshot.time = time_group.first;
        snapshot.rows = time_group.second;
        snapshot.data = read_measurement_set_rows(ms_path, snapshot.rows);

        snapshots.push_back(std::move(snapshot));
    }

    return snapshots;
}

int main_oldinfo(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: fip oldinfo /path/to/file.ms\n";
        return 1;
    }

    const std::string ms_path = argv[1];

    try {
        const auto snapshots = read_measurement_set_snapshots(ms_path);

        std::cout << "Found " << snapshots.size() << " snapshots\n";

        for (const auto& snapshot : snapshots) {
            const HostMeasurementSetData& data = snapshot.data;

            std::cout << "snapshot " << snapshot.index
                      << " time " << snapshot.time
                      << " rows " << data.num_rows
                      << " channels " << data.num_channels
                      << " samples " << data.num_samples
                      << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "Failed: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
