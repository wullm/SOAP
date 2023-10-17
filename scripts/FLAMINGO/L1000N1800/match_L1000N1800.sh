#!/bin/bash -l
#
# Match halos between 1Gpc DMO and HYDRO runs
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J match_L1000N1800_DMO_FIDUCIAL
#SBATCH -o ./logs/matching/%x_%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 1:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

set -e

sim1=DMO_FIDUCIAL
for sim2 in HYDRO_FIDUCIAL HYDRO_STRONG_AGN HYDRO_WEAK_AGN ; do

  # Snapshot to do
  snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

  # Location of the files
  basedir="/cosma8/data/dp004/flamingo/Runs/"
  vr_basename1="${basedir}/L1000N1800/${sim1}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
  vr_basename2="${basedir}/L1000N1800/${sim2}/VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
  nr_particles=10

  # Where to put the output
  outdir="/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/L1000N1800/${sim1}/matching/TO_${sim2}/"
  \mkdir -p ${outdir}
  outfile="${outdir}/match_L1000N1800_${sim1}_${sim2}_${snapnum}.field_only.n${nr_particles}.hdf5"

  echo
  echo Matching $sim1 to $sim2, snapshot ${snapnum}
  echo

  mpirun python3 -u -m mpi4py \
      ./match_vr_halos.py ${vr_basename1} ${vr_basename2} ${nr_particles} ${outfile} --use-types 1 --to-field-halos-only

done

echo All matching done.

