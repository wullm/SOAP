#!/bin/bash -l
#
# Compute VR group membership for each particle in a snapshot.
# Job name determines which of the L0100N0180 runs we process.
# Array job index is the snapshot number to do.
#
# Submit with (for example):
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=6 ./scripts/FLAMINGO/L0100N0180/group_membership_L0100N0180.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/group_membership_L1000N1800_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 0:30:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L0100N0180/${SLURM_JOB_NAME}"

# Input simulation location
basedir="/cosma8/data/dp004/flamingo/Runs/${sim}/"

# Where to write the output
outbase="/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/${sim}/SOAP_uncompressed/"

# Generate input and output file names
swift_filename="${basedir}/snapshots/flamingo_${snapnum}.hdf5"
vr_basename="${basedir}/VR/halos_${snapnum}"
outfile="${outbase}/membership_${snapnum}/membership_${snapnum}.%(file_nr)d.hdf5"

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M ${outdir}

# Run the code
mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
