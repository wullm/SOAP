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

swift_filename="/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/snapshots/flamingo_0013.hdf5"
vr_basename="/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/vr/catalogue_0013/vr_catalogue_0013"
outfile="/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/group_membership/vr_membership_0013.hdf5"

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
