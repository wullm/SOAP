#!/bin/bash -l
#
# Compute VR group membership for each particle in a snapshot.
# Output locations are specified by enviroment variables. E.g.
#
# export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/
# export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/ScienceRuns/
#
# Job name determines which of the L1000N0900 runs we process.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N0900/hbt_group_membership_L1000N0900.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=16
#SBATCH -o ./logs/group_membership_L1000N0900_%x.%a.out
#SBATCH -p cosma8-shm2
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 0:30:00
#

# Prevent numpy starting huge numbers of threads
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Get location for temporary output
if [[ "${FLAMINGO_SCRATCH_DIR}" ]] ; then
  scratch_dir="${FLAMINGO_SCRATCH_DIR}"
else
  echo Please set FLAMINGO_SCRATCH_DIR
  exit 1
fi

# Get location for final output
if [[ "${FLAMINGO_OUTPUT_DIR}" ]] ; then
  output_dir="${FLAMINGO_OUTPUT_DIR}"
else
  echo Please set FLAMINGO_OUTPUT_DIR
  exit 1
fi

# Check if this is a slurm job
if [[ -z "${SLURM_JOB_ID}" ]]; then
    # Interactive run
    if [[ $# -ne 2 ]]; then
        echo Non-slurm usage: group_membership.sh run_name snap_nr
        exit 1
    else
        sim=L1000N0900/${1}
        snapnum=`printf '%04d' ${2}`
        snapnum3=`printf '%03d' ${2}`        
        np_flag="-np 16"
    fi
else
    # Slurm job
    sim="L1000N0900/${SLURM_JOB_NAME}"    
    snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`
    snapnum3=`printf '%043' ${SLURM_ARRAY_TASK_ID}`
    np_flag=""
fi

# Input simulation location
basedir="/cosma8/data/dp004/flamingo/Runs/${sim}/"

# Where to write the output
outbase="${scratch_dir}/${sim}/HBT/SOAP_uncompressed/"

# Generate input and output file names
swift_filename="${basedir}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
hbt_basename="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/HBT/hbt/${snapnum3}/SubSnap_${snapnum3}"
outfile="${outbase}/membership_${snapnum}/membership_${snapnum}.%(file_nr)d.hdf5"

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M ${outdir}

# Run the code
mpirun ${np_flag} python3 -u -m mpi4py \
    ./group_membership.py ${swift_filename} ${hbt_basename} HBTplus ${outfile}
