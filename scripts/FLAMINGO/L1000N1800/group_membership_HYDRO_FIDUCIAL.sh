#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J test
#SBATCH -o ./test.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 0:20:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

swift_filename="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.%(file_nr)d.hdf5"
vr_basename="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077"
outfile="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/HYDRO_FIDUCIAL/group_membership/vr_membership_0077.%(file_nr)d.hdf5"

outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
