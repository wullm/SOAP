#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J test_group_membership
#SBATCH -o ./logs/test_group_membership.%a.out
#SBATCH -p cosma7-shm
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 2:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`
basedir=/cosma7/data/dp004/jch/FLAMINGO/L0200N0360/HYDRO_FIDUCIAL/

swift_filename="${basedir}/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
vr_basename="${basedir}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="${basedir}/SOAP/group_membership/vr_membership_${snapnum}.%(file_nr)d.hdf5"

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
