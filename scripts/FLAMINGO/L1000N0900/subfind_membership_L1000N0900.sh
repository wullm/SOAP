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
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N0900/subfind_group_membership_L1000N0900.sh
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

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`
snapnum3=`printf '%03d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L1000N0900/${SLURM_JOB_NAME}"

# Input simulation location
basedir="/cosma8/data/dp004/flamingo/Runs/${sim}/"

# Where to write the output
outbase="${scratch_dir}/${sim}/Subfind/SOAP_uncompressed/"

# Generate input and output file names
swift_filename="${basedir}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
#subfind_basename="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/HBT/hbt/${snapnum3}/SubSnap_${snapnum3}"
subfind_basename="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/gadget_snapshots/snapdir_${snapnum3}/snapshot_${snapnum3}"
outfile="${outbase}/membership_${snapnum}/membership_${snapnum}.%(file_nr)d.hdf5"

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M ${outdir}

# Run the code
mpirun python3 -u -m mpi4py \
    ./group_membership.py ${swift_filename} ${subfind_basename} Gadget4 ${outfile}
