#! /bin/bash
#
# Compress the membership files using h5repack, applying GZIP compression
#
# cd SOAP
# mkdir logs
# sbatch -J HYDRO_FIDUCIAL --array=6 ./scripts/FLAMINGO/L0100N0180/compress_group_membership_L0100N0180.sh
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH -o ./logs/compress_membership_L0100N0180_%x.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 01:00:00
#

module purge
module load intel_comp/2018
module load hdf5
module load gnu-parallel

# Which snapshot to do
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# Which simulation to do
sim="L0100N0180/${SLURM_JOB_NAME}"

# Location of the input to compress
inbase="/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/${sim}/SOAP_uncompressed/"

# Location of the compressed output
outbase="/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/${sim}/SOAP/"

# Create the output folder if it does not exist
outdir="${outbase}/membership_${snapnum}"
mkdir -p "${outdir}"
lfs setstripe --stripe-count=-1 --stripe-size=32M "${outdir}"

# Uncompressed membership file basename
input_filename="${inbase}/membership_${snapnum}/membership_${snapnum}"

# Compressed membership file basename
output_filename="${outbase}/membership_${snapnum}/membership_${snapnum}"

# Determine how many files we have
nr_files=`ls -1 ${input_filename}.*.hdf5 | wc -l`
nr_files_minus_one=$(( ${nr_files} - 1 ))

# run h5repack in parallel using 32 processes on files 0 to 63
# we could use more processes, but that causes a larger strain for the file
# system and is therefore not really more efficient
# make sure to update the 'seq' arguments when there are more/less membership
# files
echo Compressing ${nr_files} group membership files
echo Source     : ${input_filename}
echo Destination: ${output_filename}

seq 0 ${nr_files_minus_one} | parallel -j 32 \
  h5repack \
    -i "${input_filename}.{}.hdf5" \
    -o "${output_filename}.{}.hdf5" \
    -l CHUNK=10000 -f GZIP=9

echo Done.

