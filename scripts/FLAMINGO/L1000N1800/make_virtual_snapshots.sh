#!/bin/bash

module purge
module load python/3.10.1

set -e

# Name of the simulation to do
run="L1000N1800/${1}"
simdir="/cosma8/data/dp004/flamingo/Runs/${run}/"

# Location of the output
outdir="${2}/${run}"
\mkdir -p ${outdir}

cd ../../..

# Loop over snapshots
for snap_nr in `seq 0 77` ; do

    # Location of the input snapshot files
    snap_str=`printf '%04d' ${snap_nr}`
    echo Snapshot ${snap_str}
    snapshot="${simdir}/snapshots/flamingo_${snap_str}/flamingo_${snap_str}.%(file_nr)d.hdf5"

    # Location of the input membership files
    membership="${simdir}/SOAP/membership_${snap_str}/membership_${snap_str}.%(file_nr)d.hdf5"

    # Output file to create
    outfile="${outdir}/flamingo_${snap_str}.hdf5"

    # Run the code
    echo python3 ./make_virtual_snapshot.py "${snapshot}" "${membership}" "${outfile}"
    python3 ./make_virtual_snapshot.py "${snapshot}" "${membership}" "${outfile}"

done
