#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH -J test_halo_properties
#SBATCH -o ./logs/test_halo_properties.%a.out
#SBATCH -p cordelia
#SBATCH -A durham
##SBATCH --exclusive
#SBATCH -t 2:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`
basedir=/cosma7/data/dp004/jch/FLAMINGO/L0200N0360/HYDRO_FIDUCIAL/

swift_filename="${basedir}/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
vr_basename="${basedir}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
nr_chunks=8
scratch_dir="${basedir}/SOAP/chunks-tmp/"
outfile="${basedir}/SOAP/halo_properties/halo_properties_%(snap_nr)04d.hdf5"
extra_filename="${basedir}/SOAP/group_membership/vr_membership_%(snap_nr)04d.%(file_nr)d.hdf5"

mpirun python3 -u -m mpi4py \
    ./compute_halo_properties.py ${swift_filename} ${scratch_dir} ${vr_basename} ${outfile} ${SLURM_ARRAY_TASK_ID} \
    --chunks=${nr_chunks} \
    --extra-input=${extra_filename}
