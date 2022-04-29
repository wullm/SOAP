#!/bin/bash -l
#
# Compute halo properties for a snapshot. Must run the group_membership
# script first.
#
# Job name determines which of the L1000N1800 runs we process.
# Array job index is the snapshot number to do. Submit with (for example):
#
# sbatch -J HYDRO_FIDUCIAL --array=78 ./halo_properties_L1000N1800.sh
#
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH -J test_halo_props
#SBATCH -o ./logs/halo_properties_L1000N1800_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 4:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L1000N1800/${SLURM_JOB_NAME}"

# Input simulation location
basedir="/cosma8/data/dp004/flamingo/Runs/L1000N1800/${sim}/"

# Where to write the output
outbase="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${sim}/"

# Generate file names for this snapshot
swift_filename="${basedir}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
extra_filename="${outbase}/group_membership/vr_membership_${snapnum}.%(file_nr)d.hdf5"
vr_basename="${basedir}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="${outbase}/${sim}/halo_properties/halo_properties_${snapnum}.hdf5"

chunks_per_dimension=2

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"

mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${swift_filename} ${vr_basename} ${outfile} \
    --chunks-per-dimension=${chunks_per_dimension} \
    --extra-input=${extra_filename}
