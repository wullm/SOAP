#!/bin/bash
#
# Submit jobs to process all snapshots of a FLAMINGO run.
# Takes the simulation name and range of snapshots to do as a parameter.
# The range of snapshots is passed to the sbatch --array argument.
#
# Run from the scripts/FLAMINGO directory. E.g.
#
# cd SOAP/scripts/FLAMINGO
# ./submit_all_jobs.sh L0100N0180/HYDRO_FIDUCIAL 0-6
#
# Submits a series of array jobs with dependencies between elements.
#
# Note that if a job fails you may get jobs stuck in the queue in state
# "DependencyNeverSatisfied", which will need to be cancelled.
#

# Set default output locations
if [[ ! "${FLAMINGO_OUTPUT_DIR}" ]] ; then
  export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/SOAP-Output/
fi
if [[ ! "${FLAMINGO_SCRATCH_DIR}" ]] ; then
  export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/SOAP-Output/
fi

if [[ "$#" -lt 2 ]] ; then
  echo
  echo "Usage: ./submit_jobs.sh --run=run_name --snapshots=snapshots  \ "
  echo "           [--membership] [--soap] [--compress-soap] [--compress-membership]"
  echo
  echo "Run name should be box size and model e.g. --run=L1000N1800/HYDRO_FIDUCIAL"
  echo "Snapshots are in format used by sbatch --array option e.g. --snapshots=0-77%4"
  echo
  exit 1
fi

# Get command line arguments
do_all=1
args=$(getopt -l"run:,snapshots:,membership,soap,compress-membership,compress-soap" -o "" -- "$@")
eval set -- "$args"
while [ $# -ge 1 ]; do
  case "$1" in
    --)
      shift
      break
      ;;
    --run)
      run_name="$2"
      shift 2
      ;;
    --snapshots)
      snaps="$2"
      shift 2
      ;;
    --membership)
      do_membership=1
      do_all=0
      shift
      ;;
    --soap)
      do_soap=1
      do_all=0
      shift
      ;;
    --compress-membership)
      do_compress_membership=1
      do_all=0
      shift
      ;;
    --compress-soap)
      do_compress_soap=1
      do_all=0
      shift
      ;;
  esac
done
echo
if [[ "${do_all}" == 1 ]] ; then
  do_membership=1
  do_soap=1
  do_compress_membership=1
  do_compress_soap=1
fi

# Check we have run name and snapshots
if [[ ! "${run_name}" ]] ; then
  echo Please specify run, e.g. --run=L1000N1800/HYDRO_FIDUCIAL
  exit 1
fi
if [[ ! "${snaps}" ]] ; then
  echo Please specify snapshots to do, e.g. --snapshots=0-77%4
  exit 1
fi

# Make sure the run exists
run_dir="/cosma8/data/dp004/flamingo/Runs/${run_name}"
if [[ ! -d ${run_dir} ]] ; then
  echo No simulation directory "${run_dir}"
  echo
  echo Incorrect --run argument? Should be of form "--run=L????N????/[DMO|HYDRO]_*"
  echo E.g. --run=L1000N1800/HYDRO_FIDUCIAL
  echo
  exit 1
fi

# Get the simulation box size (L????N????) from the run name
box=`echo "${run_name}" | sed 's/\(L....N....\)\/.*/\1/'`
if [[ "$box" ]] ; then
  echo Simulation box: "${box}"
else
  echo Cannot extract box size from simulation name
  echo Name should be of form "L????N????/[DMO|HYDRO]_*"
  exit 1
fi

# Get the model (HYDRO_FIDUCIAL etc) from the run name
model=`echo "${run_name}" | sed 's/L....N....\/\(.*\)/\1/'`
if [[ "$model" ]] ; then
  echo Model: "${model}"
else
  echo Cannot extract model name from simulation name
  echo Name should be of form "L????N????/[DMO|HYDRO]_*"
  exit 1
fi

# Go to top level SOAP source dir
cd ../..
if [[ -e compute_halo_properties.py ]] ; then
  echo
  echo Submitting jobs for ${run_name} snapshots ${snaps}
  echo
else
  echo Please run this script from the SOAP/scripts/FLAMINGO directory
  exit 1
fi

# Make sure log dir exists
\mkdir -p logs

# Check that the script directory exists
script_dir="./scripts/FLAMINGO/${box}"
if [ -d "${script_dir}" ] ; then
  echo Using batch scripts from "${script_dir}" 
else
  echo No script directory "${script_dir} (maybe box size is wrong?)"
  exit 1
fi
echo

# Extra sbatch args to set output locations
extra_args="--export=ALL,FLAMINGO_OUTPUT_DIR,FLAMINGO_SCRATCH_DIR"

# Submit group membership jobs
if [[ "${do_membership}" == 1 ]] ; then
  memb_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} ${script_dir}/group_membership_${box}.sh`
  if [[ $? == 0 ]] ; then
    echo Group membership job ID is ${memb_jobid}
    # Some jobs can only run after the membership files have been created
    require_membership_files="--dependency=aftercorr:${memb_jobid}"
  else
    echo Failed to submit group membership job
    exit 1
  fi
else
  # If we're not creating membership files, assume they already exist
  require_membership_files=""
fi

# Submit halo properties jobs
if [[ "${do_soap}" == 1 ]] ; then
  props_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} ${require_membership_files} ${script_dir}/halo_properties_${box}.sh`
  if [[ $? == 0 ]] ; then
    echo Halo properties job ID is ${props_jobid}
    # Some jobs can only run after SOAP has run
    require_soap_output="--dependency=aftercorr:${props_jobid}"
  else
    echo Failed to submit halo properties job
    exit 1
  fi
else
  # If we're not creating halo properties files, assume they already exist
  require_soap_output=""  
fi

# Submit group membership compression jobs
if [[ "${do_compress_membership}" == 1 ]] ; then
  comp_memb_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} ${require_membership_files} ${script_dir}/compress_group_membership_${box}.sh`
  if [[ $? == 0 ]] ; then
    echo Membership compression job ID is ${comp_memb_jobid}
  else
    echo Failed to submit membership compression job
    exit 1
  fi
fi

# Submit halo properties compression jobs
if [[ "${do_compress_soap}" == 1 ]] ; then
  comp_props_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} ${require_soap_output} ${script_dir}/compress_halo_properties_${box}.sh`
  if [[ $? == 0 ]] ; then
    echo Properties compression job ID is ${comp_props_jobid}
  else
    echo Failed to submit properties compression job
    exit 1
  fi
fi

echo
squeue -j ${memb_jobid},${props_jobid},${comp_memb_jobid},${comp_props_jobid}

echo
echo Scratch dir: ${FLAMINGO_SCRATCH_DIR}
echo Output dir : ${FLAMINGO_OUTPUT_DIR}
echo
echo See `pwd`/logs for job output when jobs start
echo
