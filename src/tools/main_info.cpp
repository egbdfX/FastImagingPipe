#include "main.h"

#include <stdio.h>

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


using namespace casacore;


int main_info(int argc, char* argv[]){
    size_t num_snapshots, num_rows, num_channels=0;

    if(argc < 2){
        fprintf(stdout, "%s", "Usage: fip info /path/to/file.ms\n");
        return 1;
    }


    /* Open Table. */
    if(!Table::isReadable(argv[1])){
        fprintf(stderr, "Error: %s is not readable!\n", argv[1]);
        return 1;
    }
    Table ms(argv[1], Table::Old);
    Table spw       = ms.keywordSet().asTable("SPECTRAL_WINDOW");
    Table subset    = ms.tableDesc().isColumn("FLAG_ROW") ? ms(!ms.col("FLAG_ROW")) : ms;
    TableIterator iter(subset, "TIME", TableIterator::Ascending,
                                       TableIterator::QuickSort);


    /* Determine number of channels */
    ROArrayColumn<double> chan_freq_col(spw, "CHAN_FREQ");
    if(chan_freq_col.shapeColumn().empty())
        for(size_t i=0; i<chan_freq_col.nrow(); i++)
            num_channels += chan_freq_col.shape(i).product();
    else
        num_channels = (size_t)chan_freq_col.shapeColumn().product();


    /* Iterate and print snapshot details. */
    for(num_snapshots=0; !iter.pastEnd(); iter++){
        Table snapshot = iter.table();
        num_rows = (size_t)snapshot.nrow();
        if(num_rows == 0)
            continue; /* No usable rows. Skip. */

        /* Otherwise, we have a usable snapshot. */
        num_snapshots++;

        double timestamp_double = snapshot.col("TIME").getDouble(0);
        Unit   timestamp_unit   = snapshot.col("TIME").unit();
        MVTime timestamp_date   = MVTime(Quantity(timestamp_double,
                                                  timestamp_unit));
        String timestamp_string = timestamp_date.string(MVTime::ISO);

        fprintf(stdout, "snapshot %4zu @ time %s (%.3f): rows %5zu, "
                        "channels %4zu, samples %8zu\n",
                        num_snapshots-1,
                        timestamp_string.c_str(),
                        timestamp_double,
                        num_rows,
                        num_channels,
                        num_rows*num_channels);
    }
    fprintf(stdout, "TOTAL: %6zu snapshots\n", num_snapshots);
    fprintf(stdout, "\n");
    return 0;
}
