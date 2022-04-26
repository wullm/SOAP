#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J test_group_membership
#SBATCH -o ./test_group_membership.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 1:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

swift_filename="/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/snapshots/flamingo_${snapnum}.hdf5"
vr_basename="/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/vr/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/group_membership/vr_membership_${snapnum}.hdf5"

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
