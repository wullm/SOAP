#!/bin/env python

import sys
import os
import argparse
from mpi4py import MPI

class ArgumentParserError(Exception): pass

class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(message+"\n")
        raise ArgumentParserError(message)

def get_halo_props_args(comm):
    """
    Process command line arguments for halo properties program.

    Returns a dict with the argument values, or None on failure.
    """
    
    if comm.Get_rank() == 0:

        os.environ['COLUMNS'] = '80' # Can't detect terminal width when running under MPI?

        parser = ThrowingArgumentParser(description='Compute halo properties in SWIFT snapshots.')
        parser.add_argument('swift_filename',
                            help='Format string to generate snapshot filenames. Use %%(file_nr)d for the file number and %%(snap_nr)04d for the snapshot.')
        parser.add_argument('vr_basename',
                            help='Format string to generate base name of the VELOCIraptor files, excluding trailing .properties[.N] etc. Use %%(snap_nr)04d for the snapshot.')
        parser.add_argument('output_file', help='Format string to generate name of the output file. Use %%(snap_nr)04d for the snapshot.')
        parser.add_argument('snapshot_nr', help='Snapshot number to process', type=int)
        parser.add_argument("--chunks-per-dimension", metavar="N", type=int, default=1,
                            help="Splits volume in N**3 chunks and each compute node processes one chunk at a time")
        parser.add_argument("--extra-input", metavar="FORMAT_STRING",
                            help="Format string to generate names of files with additional particle datasets (e.g. halo membership). Use %%(file_nr)d for the file number and %%(snap_nr)04d for the snapshot.")
        parser.add_argument("--centrals-only", action="store_true", help="Only process central halos")
        parser.add_argument("--max-halos", metavar="N", nargs=1, type=int, default=(0,),
                            help="(For debugging) only process the first N halos in the catalogue")
        parser.add_argument("--calculations", nargs="*", help="Which calculations to do (default is to do all)")
        parser.add_argument("--reference-snapshot", help="Specify reference snapshot number containing all particle types", metavar="N", type=int)
        try:
            args = parser.parse_args()
        except ArgumentParserError as e:
            args = None

    else:
        args = None

    args = comm.bcast(args)
    if args is None:
        MPI.Finalize()
        sys.exit(0)

    return args


def get_group_membership_args(comm):
    """
    Process command line arguments for halo properties program.

    Returns a dict with the argument values, or None on failure.
    """
    
    if comm.Get_rank() == 0:

        os.environ['COLUMNS'] = '80' # Can't detect terminal width when running under MPI?

        parser = ThrowingArgumentParser(description='Compute particle group membership in SWIFT snapshots.')
        parser.add_argument('swift_filename',
                            help='Format string to generate snapshot filenames. Use %%(file_nr)d for the file number.')
        parser.add_argument('vr_basename',
                            help='Base name of the VELOCIraptor files, excluding trailing .properties[.N] etc.')
        parser.add_argument('output_file', help='Format string to generate output filenames. Use %%(file_nr)d for the file number.')

        try:
            args = parser.parse_args()
        except ArgumentParserError as e:
            args = None

    else:
        args = None

    args = comm.bcast(args)
    if args is None:
        MPI.Finalize()
        sys.exit(0)

    return args

