#!/bin/bash -l
#
# Compute VR group membership for each particle in a snapshot.
# Job name determines which of the L1000N1800 runs we process.
# Array job index is the snapshot number to do.
#
# This version takes particles in one simulation and computes their
# group mmebership in a VR output from another simulation.
# Used to investigate halo matching between dmo and hydro.
#
# Submit with (for example):
#
# sbatch -J HYDRO_FIDUCIAL --array=78 ./group_membership_L1000N1800.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J cross_match
#SBATCH -o ./logs/group_membership_L1000N1800_%x.%a.out
#SBATCH -p cosma8-shm
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 1:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulations to do
box="L1000N1800"
sim1="DMO_FIDUCIAL"
sim2="HYDRO_FIDUCIAL"

basedir1="/cosma8/data/dp004/flamingo/Runs/${box}/${sim1}/"
basedir2="/cosma8/data/dp004/flamingo/Runs/${box}/${sim2}/"

outbase1="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${box}/${sim1}/"
outbase2="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/${box}/${sim2}/"

swift_filename1="${basedir1}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
swift_filename2="${basedir2}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"

vr_basename1="${basedir1}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
vr_basename2="${basedir2}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"

outfile1="${outbase1}/cross_membership/group_membership_${sim2}_${snapnum}/vr_membership_${snapnum}.%(file_nr)d.hdf5"
outfile2="${outbase2}/cross_membership/group_membership_${sim1}_${snapnum}/vr_membership_${snapnum}.%(file_nr)d.hdf5"

vfile1="${outbase1}/group_membership/group_membership_${snapnum}/flamingo_${snapnum}.hdf5"
vfile2="${outbase2}/group_membership/group_membership_${snapnum}/flamingo_${snapnum}.hdf5"

# For particles in sim1, find VR membership in sim2
outdir=`dirname "${outfile1}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"
mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename1} ${vr_basename2} ${outfile1} --update-virtual-file=${vfile1} --output-prefix=${sim2}_

# For particles in sim2, find VR membership in sim1
outdir=`dirname "${outfile2}"`
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"
mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename2} ${vr_basename1} ${outfile2} --update-virtual-file=${vfile2} --output-prefix=${sim1}_
