#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J test_halo_properties
#SBATCH -o ./logs/test_halo_properties.%a.out
#SBATCH -p cosma
#SBATCH -A durham
##SBATCH --exclusive
#SBATCH -t 2:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

basedir="/cosma5/data/jch/HaloProperties/200_w_lightcone/"

swift_filename="${basedir}/snapshots/flamingo_%(snap_nr)04d.hdf5"
vr_basename="${basedir}/vr/catalogue_%(snap_nr)04d/vr_catalogue_%(snap_nr)04d"
nr_chunks=1
outfile="${basedir}/halo_properties/halo_properties_%(snap_nr)04d.hdf5"
extra_filename="${basedir}/group_membership/vr_membership_%(snap_nr)04d.hdf5"

mpirun python3 -u -m mpi4py \
    ./compute_halo_properties.py ${swift_filename} ${vr_basename} ${outfile} ${SLURM_ARRAY_TASK_ID} \
    --chunks=${nr_chunks} \
    --extra-input=${extra_filename}
