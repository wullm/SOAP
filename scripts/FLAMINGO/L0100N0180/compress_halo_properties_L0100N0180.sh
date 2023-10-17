#!/bin/bash
#
# Compress SOAP catalogues.
#
# Output locations are specified by enviroment variables. E.g.
#
# export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/
# export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/ScienceRuns/
#
# To run:
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=6 ./scripts/FLAMINGO/L0100N0180/compress_halo_properties_L0100N0180.sh
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/compress_properties_L0100N0180_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 01:00:00
#

module purge
module load python/3.10.1

# Get location for temporary output
if [[ "${FLAMINGO_SCRATCH_DIR}" ]] ; then
  scratch_dir="${FLAMINGO_SCRATCH_DIR}"
else
  echo Please set FLAMINGO_SCRATCH_DIR
  exit 1
fi

# Get location for final output
if [[ "${FLAMINGO_OUTPUT_DIR}" ]] ; then
  output_dir="${FLAMINGO_OUTPUT_DIR}"
else
  echo Please set FLAMINGO_OUTPUT_DIR
  exit 1
fi

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L0100N0180/${SLURM_JOB_NAME}"

# compression script
script="./compression/compress_fast_metadata.py"

# Location of the input to compress
inbase="${scratch_dir}/${sim}/SOAP_uncompressed/"

# Location of the compressed output
outbase="${output_dir}/${sim}/SOAP/"

# Name of the input SOAP catalogue
input_filename="${inbase}/halo_properties_${snapnum}.hdf5"

# name of the output SOAP catalogue
output_filename="${outbase}/halo_properties_${snapnum}.hdf5"

# directory used to store temporary files (preferably a /snap8 directory for
# faster writing and reading)
scratch_dir="${scratch_dir}/${sim}/SOAP_compression_tmp/"

# run the script using all available threads on the node
python3 ${script} --nproc 128 ${input_filename} ${output_filename} ${scratch_dir}
