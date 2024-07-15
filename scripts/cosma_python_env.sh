#!/bin/bash
#
# Set up a virtual env suitable for running mpi4py code using openmpi and
# parallel HDF5.
#

set -e

hdf5_version=1.12.3

module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/${hdf5_version}
python -m pip cache purge

# Name of the new venv to create
venv_name="openmpi-5.0.3-hdf5-${hdf5_version}-env"

# Create an empty venv
python -m venv "${venv_name}"

# Activate the venv
source "${venv_name}"/bin/activate

# Install mpi4py
pip install mpi4py

# Install h5py
export CC="`which mpicc`"
export HDF5_MPI="ON"
export HDF5_DIR=${HDF5_HOME}
pip install setuptools cython "numpy<2" pkgconfig
pip install --no-binary h5py --no-build-isolation h5py

# Add symlink to the right mpirun in the venv's bin directory
mpirun=`which mpirun`
ln -s "${mpirun}" "${venv_name}"/bin/mpirun

# Install other modules
pip install -r requirements.txt
git clone https://github.com/jchelly/VirgoDC.git "${venv_name}/VirgoDC"
pip install "${venv_name}/VirgoDC/python"
