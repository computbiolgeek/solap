#!/usr/bin/env python3

from Bio.PDB import PDBParser
from argparse import ArgumentParser
import os
import logging

# module level program run-time logs
logging.basicConfig(
    level=logging.INFO
)


def parse_cmd_args():
    """Parse command line arguments.

    Returns
    -------
    Namespace
        A Namespace object built up from attributes parsed out of the
        command line.

    """
    parser = ArgumentParser(description='Get alpha-carbon Cartesian '
                            'coordinates for the given list of residues.')
    parser.add_argument('--input', '-i', dest='input', type=str,
                        required=True, help='A list of residue sequence '
                        'positions.')
    parser.add_argument('--output', '-o', dest='output', type=str,
                        required=True, help='File to which to write the '
                        'Cartesian coordinates.')
    parser.add_argument('--pdb', '-p', dest='pdb', type=str, required=True,
                        help='PDB file.')
    return parser.parse_args()


def main():
    """Extract the x, y coordinates for a list of residues from the given
    PDB file on the command line.

    The residues are specified by their residue sequence positions in a
    file given on the command line. One position per line. A PDB file from
    which to extract coordinates is required on the command line. The
    extracted x, y coordinates are written to the output file in the CSV
    format. One pair of coordinates per line.

    """
    # parse command-line arguments
    logging.info('Parsing command-line arguments ...')
    args = parse_cmd_args()

    # read in the list of residue IDs
    logging.info('Reading in the list of residue sequence positions from %s'
                 % args.input)
    with open(args.input, 'rt') as ipf:
        res_ids = [int(i.strip()) for i in ipf]

    # parse the PDB file
    logging.info('Parsing the PDB file %s' % args.pdb)
    pdb_parser = PDBParser()
    pdb_id = os.path.basename(args.pdb).split('.')[0]
    structure = pdb_parser.get_structure(id=pdb_id, file=args.pdb)
    model = structure[0]

    # get x, y coordinates of alpha carbon for each residue
    logging.info('Extracting x, y coordinates ...')
    xy_coords = []
    for i in res_ids:
        for r in model.get_residues():
            if r.get_id()[1] == i:
                xy_coords.append(r['CA'].coord[:2])

    # write to file
    logging.info('Writing extracted coordinates to %s' % args.output)
    with open(args.output, 'wt') as opf:
        all_coords_str = ['%.2f, %.2f' % tuple(coord) for coord in xy_coords]
        opf.write('\n'.join(all_coords_str))


if __name__ == '__main__':
    main()