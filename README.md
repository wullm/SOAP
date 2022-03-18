# Program to compute halo properties in SWIFT simulations

This is an MPI parallel python program to compute halo properties.
The simulation volume is split into chunks. Each compute node reads
in the particles in one chunk at a time and calculates the properties
of all halos in that chunk.

Within a compute node there is one MPI process per core. The particle
data and halo catalogue for the chunk are stored in shared memory.
Each core claims a halo to process, locates the particles in a sphere
around the halo, and calculates the required properties. When all halos
in the chunk have been done the compute node will move on to the next
chunk.

## Usage on COSMA

Load modules:
```
module load python/3.10.1 gnu_comp/11.1.0 openmpi/4.1.1
```
Set parameters
```
# Format string to generate snapshot file names
swift_filename="./flamingo_0078/flamingo_0078.%(file_nr)d.hdf5"

# Name of the VR .properties file, excluding trailing .N
vr_basename="./snapshots/VR/catalogue_0078/vr_catalogue_0078.properties"

# Simulation volume is split into chunks_per_dimension^3 pieces
chunks_per_dimension=3

# Where to write the output
outfile=./output.hdf5
```
To run the code:
```
mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${swift_filename} ${vr_basename} ${chunks_per_dimension} ${outfile}
```

## Adding quantities

Property calculations are defined as classes in halo_properties.py. See
halo_properties.SOMasses for an example. Each class should have the following
attributes:

  * particle_properties - specifies which particle properties should be read in. This is a dict with one entry per particle type named "PartType0", "PartType1" etc. Each entry is a list of the names of the properties needed for that particle type.
  * mean_density_multiple - specifies that particles must be read in a sphere of mean density no greater than this multiple of the mean density
  * critical_density_multiple - specifies that particles must be read in a sphere of mean density no greater than this multiple of the critical density

There should also be a __call__ method which implements the calculation
and returns a dict with the calculated properties.

New classes must be added to halo_prop_list in compute_halo_properties.py.

## TODO

Possible improvements:

  * Compute or read in VR halo membership to compute sums over bound particles only
  * More flexible domain decomposition (e.g. Gadget style space filling curve)
  * Assign initial search radii to halos individually and repeat part of calculation if too small
  * Compute cells to read halo by halo instead of just using the bounding box
