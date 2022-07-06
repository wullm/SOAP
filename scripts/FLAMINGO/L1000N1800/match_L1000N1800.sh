#!/bin/bash -l
#
# Match halos between 1Gpc DMO and HYDRO runs
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J match_L1000N1800_DMO_FIDUCIAL_HYDRO_FIDUCIAL
#SBATCH -o ./logs/%x_%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
##SBATCH --exclusive
#SBATCH -t 12:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Simulations to match between
sim1=DMO_FIDUCIAL
sim2=HYDRO_FIDUCIAL

# Snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Location of the files
basedir="/cosma8/data/dp004/flamingo/Runs/"
vr_basename1="${basedir}/L1000N1800/${sim1}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
vr_basename2="${basedir}/L1000N1800/${sim2}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
nr_particles=10

# Where to put the output
outfile="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/L1000N1800/${sim1}/match_L1000N1800_${sim1}_${sim2}_${snapnum}.hdf5"

mpirun python3 -u -m mpi4py \
    ./match_vr_halos.py ${vr_basename1} ${vr_basename2} ${nr_particles} ${outfile} --use-types 1
