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

# Set output locations
export FLAMINGO_OUTPUT_DIR=/cosma8/data/dp004/${USER}/FLAMINGO/ScienceRuns/
export FLAMINGO_SCRATCH_DIR=/snap8/scratch/dp004/${USER}/FLAMINGO/ScienceRuns/

# Get command line args
if [ "$#" -ne 2 ]; then
    echo "Usage: submit_all_jobs.sh <run_name> <snapshots>"
    echo
    echo "run_name: name of simulation, e.g. L1000N1800/HYDRO_FIDUCIAL"
    echo "snapshots: range of snapshots to do (e.g. 0-6)"
    echo
    exit 1
fi
run_name="${1}"
snaps="${2}"
echo

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
memb_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} ${script_dir}/group_membership_${box}.sh`
if [[ $? == 0 ]] ; then
  echo Group membership job ID is ${memb_jobid}
else
  echo Failed to submit group membership job
  exit 1
fi

# Submit halo properties jobs
props_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} --dependency=aftercorr:${memb_jobid} ${script_dir}/halo_properties_${box}.sh`
if [[ $? == 0 ]] ; then
  echo Halo properties job ID is ${props_jobid}
else
  echo Failed to submit halo properties job
  exit 1
fi

# Submit group membership compression jobs
comp_memb_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} --dependency=aftercorr:${props_jobid} ${script_dir}/compress_group_membership_${box}.sh`
if [[ $? == 0 ]] ; then
  echo Membership compression job ID is ${comp_memb_jobid}
else
  echo Failed to submit membership compression job
  exit 1
fi

# Submit halo properties compression jobs
comp_props_jobid=`sbatch --parsable ${extra_args} -J ${model} --array=${snaps} --dependency=aftercorr:${comp_memb_jobid} ${script_dir}/compress_halo_properties_${box}.sh`
if [[ $? == 0 ]] ; then
  echo Properties compression job ID is ${comp_props_jobid}
else
  echo Failed to submit properties compression job
  exit 1
fi

echo
squeue -j ${memb_jobid},${props_jobid},${comp_memb_jobid},${comp_props_jobid}

echo
echo Scratch dir: ${FLAMINGO_SCRATCH_DIR}
echo Output dir : ${FLAMINGO_OUTPUT_DIR}
echo
echo See `pwd`/logs for job output when jobs start
echo
