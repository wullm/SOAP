#!/bin/bash -l
#
# Match halos between 1Gpc DMO and HYDRO runs
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J hbt_match_L1000N1800_HYDRO_FIDUCIAL
#SBATCH -o ./logs/%x_%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
##SBATCH --exclusive
#SBATCH -t 12:00:00
#

module purge
module load gnu_comp/13.1.0
module load openmpi/4.1.4
module load parallel_hdf5/1.12.0
module load python/3.12.4

# Simulations to match between
sim1=HYDRO_FIDUCIAL
sim2=DMO_FIDUCIAL

# Snapshot to do
snapnum1='077'
snapnum2='077'

soapdir="/cosma8/data/dp004/dc-elbe1/FLAMINGO/soap3/SOAP/"

# Location of the files
basedir="/cosma8/data/dp004/flamingo/Runs/"
vr_basename1="${basedir}/L1000N1800/${sim1}/HBT/${snapnum1}/SubSnap_${snapnum1}"
vr_basename2="${basedir}/L1000N1800/${sim2}/HBT/${snapnum2}/SubSnap_${snapnum2}"
nr_particles=10

# Where to put the output
outfile="/cosma8/data/dp004/dc-elbe1/FLAMINGO/matching/hbt_match_L1000N1800_${sim1}_${snapnum1}_${sim2}_${snapnum2}.field_only.n${nr_particles}.hdf5"

mpirun python3 -u -m mpi4py \
    ${soapdir}/match_hbt_halos.py ${vr_basename1} ${vr_basename2} ${nr_particles} ${outfile} --min-particle-id 0 --max-particle-id 5832000000 --to-field-halos-only
