#!/bin/bash -l
#
# Basic test of VR halo matching:
# Try to match a snapshot to itself!
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J test_halo_matching
#SBATCH -o ./logs/test_halo_matching.%a.out
#SBATCH -p cosma
#SBATCH -A durham
##SBATCH --exclusive
#SBATCH -t 2:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

basedir="/cosma5/data/jch/HaloProperties/200_w_lightcone/"

vr_basename="${basedir}/vr/catalogue_0013/vr_catalogue_0013"
outfile="${basedir}/halo_matching/halo_matching_0013.hdf5"
nr_particles=10

mpirun python3 -u -m mpi4py \
    ./match_vr_halos.py ${vr_basename} ${vr_basename} ${nr_particles} ${outfile}
