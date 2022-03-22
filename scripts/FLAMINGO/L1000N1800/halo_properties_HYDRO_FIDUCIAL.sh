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

snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`
basedir="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/"
outbase="/cosma8/data/dp004/jch/FLAMINGO/ScienceRuns/"

swift_filename="${basedir}/snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
vr_basename="${basedir}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="${outbase}/L1000N1800/HYDRO_FIDUCIAL/halo_properties/halo_properties_${snapnum}.hdf5"

chunks_per_dimension=2

# Create output directory
outdir=`dirname "${outfile}"`
mkdir -p "${outdir}"

# Set striping (C8 default of putting first part of file on metadata server is very slow sometimes)
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"

mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${swift_filename} ${vr_basename} ${chunks_per_dimension} ${outfile}
