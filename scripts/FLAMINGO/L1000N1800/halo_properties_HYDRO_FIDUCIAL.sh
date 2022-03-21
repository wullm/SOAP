#!/bin/bash -l

#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH -J test_halo_props
#SBATCH -o ./test_halo_props.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH --reservation=jlvc76_53

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

export HDF5_USE_FILE_LOCKING=FALSE

swift_filename="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.%(file_nr)d.hdf5"
vr_basename=/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/catalogue_0077/vr_catalogue_0077.properties
chunks_per_dimension=2
outfile=/cosma8/data/dp004/jch/FLAMINGO/tmp/test.hdf5

mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${swift_filename} ${vr_basename} ${chunks_per_dimension} ${outfile}
