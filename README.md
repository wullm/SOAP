# Halo properties in SWIFT simulations

This repository contains two programs which can be used to compute extra
properties of VELOCIraptor halos in SWIFT snapshots.

These are both written in python and use mpi4py for parallelism.

## Computing halo membership for particles in the snapshot

The first program, vr_group_membership.py, can compute bound and unbound 
VELOCIraptor halo indexes for all particles in a snapshot. The output
consists of the same number of files as the snapshot with particle halo
indexes written out in the same order as the snapshot.

## Computing halo properties

The second program, compute_halo_properties.py, reads the simulation
snapshot and the output from vr_group_membership.py and uses it to
calculate halo properties. It works as follows:

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

### Required modules

The same MPI module which was used to compile mpi4py must be loaded:
```
module load python/3.10.1 gnu_comp/11.1.0 openmpi/4.1.1
```

### Calculating particle group membership

The group membership program needs the location of the snapshot file(s),
the location of the VELOCIraptor catalogue and the name of the output file(s)
to generate. For example:
```
snapnum=0077
swift_filename="./snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
vr_basename="./VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="./group_membership/vr_membership_${snapnum}.%(file_nr)d.hdf5"

mpirun python3 -u -m mpi4py \
    ./vr_group_membership.py ${swift_filename} ${vr_basename} ${outfile}
```

See scripts/FLAMINGO/L1000N1800/group_membership_L1000N1800.sh for an example
batch script.

### Calculating halo properties

To calculate halo properties:

```
swift_filename="./snapshots/flamingo_${snapnum}/flamingo_${snapnum}.%(file_nr)d.hdf5"
extra_filename="./group_membership/vr_membership_${snapnum}.%(file_nr)d.hdf5"
vr_basename="./VR/catalogue_${snapnum}/vr_catalogue_${snapnum}"
outfile="./halo_properties/halo_properties_${snapnum}.hdf5"
chunks_per_dimension=2

mpirun python3 -u -m mpi4py ./compute_halo_properties.py \
    ${swift_filename} ${vr_basename} ${outfile} \
    --chunks-per-dimension=${chunks_per_dimension} \
    --extra-input=${extra_filename}
```

Here, chunks_per_dimension determines how many chunks the simulation box is
split into. Ideally it should be set such that one chunk fills a compute node.

See scripts/FLAMINGO/L1000N1800/halo_properties_L1000N1800.sh for an example
batch script.

## Adding quantities

Property calculations are defined as classes in halo_properties.py. See
halo_properties.SOMasses for an example. Each class should have the following
attributes:

  * particle_properties - specifies which particle properties should be read in. This is a dict with one entry per particle type named "PartType0", "PartType1" etc. Each entry is a list of the names of the properties needed for that particle type.
  * mean_density_multiple - specifies that particles must be read in a sphere of mean density no greater than this multiple of the mean density
  * critical_density_multiple - specifies that particles must be read in a sphere of mean density no greater than this multiple of the critical density

There should also be a `calculate` method which implements the calculation
and updates the halo_results dict with the calculated properties. The returned
values must be unyt_arrays or unyt_quantities.

New classes must be added to halo_prop_list in compute_halo_properties.py.

## Units

All particle data are stored in unyt arrays internally. On opening the snapshot
a unyt UnitSystem is defined which corresponds to the simulation units. When
particles are read in unyt arrays are created with units based on the
attributes in the snapshot. These units are propagated through the halo
property calculations and used to write the unit attributes in the output.

Comoving quantities are handled by defining a dimensionless unit corresponding
to the expansion factor a.

## TODO

Possible improvements:

  * More flexible domain decomposition (e.g. Gadget style space filling curve)
  * Assign initial search radii to halos individually and repeat part of calculation if too small
  * Compute cells to read halo by halo instead of just using the bounding box
  * Specify multi-file inputs/outputs more consistently
  * Use swiftsimio cosmo_arrays (may require a more complete wrapping of unyt_array).
  * Specify on the command line which halo property calculations to do