#!/bin/bash
#
# Submit jobs to process all snapshots of L0100N0180 run.
# Takes the simulation name and range of snapshots to do as a parameter.
# The range of snapshots is passed to the sbatch --array argument.
#
# E.g.
#
# cd SOAP
# mkdir logs
# ./scripts/FLAMINGO/L0100N0180/submit_all_jobs.sh HYDRO_FIDUCIAL 0-6
#
# Submits a series of array jobs with dependencies between elements.
#
# Note that if a job fails you may get jobs stuck in the queue in state
# "DependencyNeverSatisfied", which will need to be cancelled.
#
#

if [ "$#" -ne 2 ]; then
    echo "Usage: submit_all_jobs.sh <run_name> <snapshots>"
    echo
    echo "run_name: name of simulation box, e.g. HYDRO_FIDUCIAL"
    echo "snapshots: range of snapshots to do (e.g. 0-6)"
    echo
    exit 1
fi

name="$1"
snaps="$2"
echo Submitting jobs for box ${name} snapshots ${snaps}

# Submit group membership jobs
memb_jobid=`sbatch --parsable -J ${name} --array=${snaps} ./scripts/FLAMINGO/L0100N0180/group_membership_L0100N0180.sh`
echo Group membership job ID is ${memb_jobid}

# Submit halo properties jobs
props_jobid=`sbatch --parsable -J ${name} --array=${snaps} --dependency=aftercorr:${memb_jobid} ./scripts/FLAMINGO/L0100N0180/halo_properties_L0100N0180.sh`
echo Halo properties job ID is ${props_jobid}

# Submit group membership compression jobs
comp_memb_jobid=`sbatch --parsable -J ${name} --array=${snaps} --dependency=aftercorr:${props_jobid} ./scripts/FLAMINGO/L0100N0180/compress_group_membership_L0100N0180.sh`
echo Membership compression job ID is ${comp_memb_jobid}

# Submit halo properties compression jobs
comp_props_jobid=`sbatch --parsable -J ${name} --array=${snaps} --dependency=aftercorr:${comp_memb_jobid} ./scripts/FLAMINGO/L0100N0180/compress_halo_properties_L0100N0180.sh`
echo Properties compression job ID is ${comp_props_jobid}
