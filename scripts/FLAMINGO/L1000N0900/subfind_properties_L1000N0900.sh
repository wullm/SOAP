#!/bin/bash -l
#
# Compute halo properties for a snapshot. Must run the group_membership
# script first.
#
# Output locations are specified by enviroment variables. E.g.
#
# export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/
# export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/ScienceRuns/
#
# Job name determines which of the L1000N0900 runs we process.
# Array job index is the snapshot number to do. Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=0-77%4 ./scripts/FLAMINGO/L1000N0900/halo_properties_L1000N0900_HYDRO.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=128
#SBATCH -o ./logs/halo_properties_L1000N0900_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 4:00:00
#

if [[ -z "${SLURM_JOB_ID}" ]]; then
    # Not a slurm job, so run in serial in debugger
    model=${1}
    snap_nr=${2}
    command="python3 -m pdb"
    extra_flags="--max-halos 1"
else
    model=${SLURM_JOB_NAME}
    snap_nr=${SLURM_ARRAY_TASK_ID}
    command="mpirun python3 -u -m mpi4py"
    extra_flags=""
fi
    
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

# Which simulation to do
sim="L1000N0900/${model}"

# Input simulation location
basedir="/cosma8/data/dp004/flamingo/Runs/${sim}/"

# Where to write the uncompressed output
outbase="${scratch_dir}/${sim}/Subfind/SOAP_uncompressed/"

# Location for temporary chunk output
chunkdir="${scratch_dir}/SOAP-tmp-Subfind/${sim}/"

# Generate file names for this snapshot
swift_filename="${basedir}/snapshots/flamingo_%(snap_nr)04d/flamingo_%(snap_nr)04d.%(file_nr)d.hdf5"
extra_filename="${outbase}/membership_%(snap_nr)04d/membership_%(snap_nr)04d.%(file_nr)d.hdf5"
#subfind_basename="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/HBT/hbt/%(snap_nr)03d/SubSnap_%(snap_nr)03d"
subfind_basename="/snap8/scratch/dp004/jch/FLAMINGO/ScienceRuns/${sim}/gadget_snapshots/snapdir_%(snap_nr)03d/snapshot_%(snap_nr)03d"
outfile="${outbase}/halo_properties_%(snap_nr)04d.hdf5"

# Check for DMO run
dmo_flag=""
if [[ $sim == DMO_* ]] ; then
  dmo_flag="--dmo"
fi

nr_chunks=1

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M ${outdir}

# Make argument list
soap_args="${swift_filename} ${chunkdir} ${subfind_basename} Gadget4 ${outfile} ${snap_nr} --chunks=${nr_chunks} ${dmo_flag} --extra-input=${extra_filename} ${extra_flags}"

# Run the code
${command} ./compute_halo_properties.py ${soap_args}

