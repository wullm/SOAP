#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J groups_L1000N1800_HYDRO_FIDUCIAL
#SBATCH -o ./logs/%x.%A.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 4:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`
basedir="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/"
outbase="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/"

swift_filename="${basedir}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.0.hdf5"
vr_basename="${basedir}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="${outbase}/L1000N1800/HYDRO_FIDUCIAL/group_membership/vr_membership_${snapnum}.%(file_nr)d.hdf5"

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"

# Set striping (C8 default of putting first part of file on metadata server is very slow sometimes)
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
