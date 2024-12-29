# losses module

import numpy as np

import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir + '/alphafold')

from alphafold.common import protein
# dssp loss imports
from Bio.PDB.DSSP import DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
# to run tmalign
import subprocess
from scipy import linalg
from jax.nn import softmax

from bigmhc import *


el_args, el_models = bigmhc_el_modelsetup()
im_args, im_models = bigmhc_im_modelsetup()

######################################################
# COMMON FUNCTIONS USED BY DIFFERENT LOSSES.
######################################################

def get_coord(atom_type, neo_object):
    '''
    General function to get the coordinates of an atom type in a pdb. For geometric-based losses.
    Returns an array [[chain, resid, x, y, z]]
    '''
    coordinates = []
    pdb_lines = protein.to_pdb(neo_object.try_unrelaxed_structure).split('\n')
    for l in pdb_lines: # parse PDB lines and extract atom coordinates
        if 'ATOM' in l and atom_type in l:
            s = l.split()
            if len(s[4]) > 1: # residue idx and chain id are no longer space-separated at high id values
                coordinates.append([s[4][0], int(s[4][1:]), np.array(s[5:8], dtype=float)])
            else:
                coordinates.append([s[4], int(s[5]), np.array(s[6:9], dtype=float)])

    coord = np.array(coordinates, dtype=object)

    # Find chain breaks.
    ch_breaks = np.where(np.diff(coord[:, 1]) > 1)[0]
    ch_ends = np.append(ch_breaks, len(coord) - 1)
    ch_starts = np.insert(ch_ends[:-1], 0, 0)

    chain_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for k, start_finish in enumerate(list(zip(ch_starts, ch_ends))):
        coord[start_finish[0] + 1 : start_finish[1]+1 , 0] = chain_list[k] # re-assign chains based on chain breaks

    return coord


def dssp_wrapper(pdbfile):
    '''Compute DSSP string on structure.'''

    dssp_tuple = dssp_dict_from_pdb_file(pdbfile, DSSP="lm_bin/dssp")[0]

    dssp_list = []
    for key in dssp_tuple.keys():
        dssp_list.append(dssp_tuple[key][2])

    return dssp_list

def calculate_dssp_fractions(dssp_list):
    '''Compute DSSP fraction based on a DSSP list.'''

    N_residues = len(dssp_list)
    fraction_beta  = float(dssp_list.count("E") ) / float(N_residues)
    fraction_helix = float(dssp_list.count("H") ) / float(N_residues)
    fraction_other = float(1.0 - fraction_beta-fraction_helix)
    # print(dssp_list, fraction_beta, fraction_helix, fraction_other)

    return fraction_beta, fraction_helix, fraction_other


def tmalign_wrapper(template, temp_pdbfile, force_alignment=None):
    if force_alignment == None:
        p = subprocess.Popen(f'lm_bin/TMalign {template} {temp_pdbfile} | grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    else:
        p = subprocess.Popen(f'lm_bin/TMalign {template} {temp_pdbfile} -I {force_alignment} | grep -E "RMSD|TM-score=" ', stdout=subprocess.PIPE, shell=True)
    output, __ = p.communicate()
    tm_rmsd  = float(str(output)[:-3].split("RMSD=")[-1].split(",")[0] )
    tm_score = float(str(output)[:-3].split("TM-score=")[-1].split("(if")[0] )

    return tm_rmsd, tm_score


############################
# LOSS COMPUTATION
############################

def compute_loss(losses, neo, args, loss_weights):
    '''
    Compute the loss of a single neoantigen.
    losses: list of list of losses and their associated arguments (if any).
    neo: a neoantigen object.
    args: the whole argument namespace (some specific arguments are required for some specific losses).
    loss_weights: list of weights associated with each loss.
    '''
    # intialize scores
    scores = []
    # iterate over all losses
    for loss_idx, current_loss in enumerate(losses) :
        loss_type, loss_params  = current_loss # assign loss and its arguments if any.

        if loss_type == 'plddt':
            # NOTE:
            # Using this loss will optimise plddt (predicted lDDT) for the sequence(s).
            score = 1. - np.mean(neo.try_prediction_results['plddt'])


        elif loss_type == 'ptm':
            # NOTE:
            # Using this loss will optimise ptm (predicted TM-score) for the sequence(s).
            score = 1. - np.mean(neo.try_prediction_results['ptm'])


        elif loss_type == 'bigmhc_el':
            score = 1. - bigmhc_el_analyze(neo.try_sequence, neo.hla, el_args, el_models)


        elif loss_type == 'bigmhc_im':
            score = 1. - bigmhc_im_analyze(neo.try_sequence, neo.hla, im_args, im_models)


        elif loss_type == 'pae':
            # NOTE:
            # Using this loss will optimise the mean of the pae matrix (predicted alignment error).
            norm = np.mean(neo.init_prediction_results['predicted_aligned_error'])
            score = np.mean(neo.try_prediction_results['predicted_aligned_error']) / norm


        elif loss_type == 'entropy':
            # CAUTION:
            # This loss is unlikely to yield anything useful at present.
            # i,j pairs that are far away from each other, or for which AF2 is unsure, have max prob in the last bin of their respective distograms.
            # This will generate an artifically low entropy for these positions.
            # Need to find a work around this issue before using this loss.
            print('Entropy definition is most likley improper for loss calculation. Use at your own risk...')

            # Distogram from AlphaFold2 represents pairwise Cb-Cb distances, and is outputted as logits.
            probs = softmax(neo.try_prediction_results['distogram']['logits'], -1. ) # convert logit to probs

            score = np.mean(-np.sum((np.array(probs) * np.log(np.array(probs))), axis=-1))


        elif loss_type == 'dual':
            # NOTE:
            # This loss jointly optimises ptm and plddt (equal weights).
            score = 1. - (np.mean(neo.try_prediction_results['plddt']) / 2.) - (neo.try_prediction_results['ptm'] / 2.)


        elif loss_type == "tmalign":
            # TODO:
            # a loss to enforce tmscore against a given template.

            # write temporary pdbfile to compute tmscore.
            temp_pdbfile = f'{args.out}_models/tmp.pdb'
            with open( temp_pdbfile , 'w') as f:
                f.write( protein.to_pdb(neo.try_unrelaxed_structure) )

            force_alignment = None

            tm_rmsd, tm_score = tmalign_wrapper(args.template, temp_pdbfile, args.template_alignment)
            print("   tm_RMSD, tmscore " , tm_rmsd, tm_score)

            score = 1. - tm_score


        elif loss_type == "dual_tmalign":
            # TODO:
            # This loss jointly optimises plddt, ptm, and tmscore against a template (equal weights).

            # write temporary pdbfile to compute tmscore.
            temp_pdbfile = f'{args.out}_models/tmp.pdb'
            with open( temp_pdbfile , 'w') as f:
                f.write( protein.to_pdb(neo.try_unrelaxed_structure) )

            tm_rmsd, tm_score = tmalign_wrapper(args.template, temp_pdbfile, args.template_alignment)
            print("   tm_RMSD, tmscore" , tm_rmsd, tm_score)

            score = 1. - (np.mean(neo.try_prediction_results['plddt']) / 3.) - (neo.try_prediction_results['ptm'] / 3.) - (tm_score / 3.)


        elif loss_type == "frac_dssp":
            # TODO:
            # a loss to enforce a fraction of secondary structure elements (e.g. 80% of fold must be beta sheet) is enforced

            # write temporary pdbfile for computing DSSP.
            temp_pdbfile = f'{args.out}_models/tmp.pdb'
            with open( temp_pdbfile , 'w') as f:
                f.write( protein.to_pdb(neo.try_unrelaxed_structure) )

            dssp_list = dssp_wrapper(temp_pdbfile)
            fraction_beta, fraction_helix, fraction_other = calculate_dssp_fractions(dssp_list)
            print (f" fraction E|H|notEH: {fraction_beta:2.2f} | {fraction_helix:2.2f} | {fraction_other:2.2f}")
            
            measured_fractions = {"E" : fraction_beta, "H" : fraction_helix, "L": fraction_other }
            
            # DSSP assigns eight states: 310-helix (represented by G), alpha-helix (H), pi-helix (I), helix-turn (T), extended beta sheet (E), beta bridge (B), bend (S) and other/loop (L).

            frac_delta = []
            for specified_secstruc in args.dssp_fractions_specified.keys() :
                if args.dssp_fractions_specified[specified_secstruc] != None:
                    frac_delta.append( np.abs( 
                        args.dssp_fractions_specified[specified_secstruc] - measured_fractions[specified_secstruc] ) 
                     ) 

            score = 0. + np.mean(frac_delta)   

        elif loss_type == "min_frac_dssp":
            # TODO:
            # a loss to enforce a minium fraction of secondary structure elements (e.g. loss is maximal if more than 80% of fold must be beta sheet) is enforced

            # write temporary pdbfile for computing DSSP.
            temp_pdbfile = f'{args.out}_models/tmp.pdb'
            with open( temp_pdbfile , 'w') as f:
                f.write( protein.to_pdb(neo.try_unrelaxed_structure) )

            dssp_list = dssp_wrapper(temp_pdbfile)
            fraction_beta, fraction_helix, fraction_other = calculate_dssp_fractions(dssp_list)
            print (f" fraction E|H|notEH: {fraction_beta:2.2f} | {fraction_helix:2.2f} | {fraction_other:2.2f}")
            
            measured_fractions = {"E" : fraction_beta, "H" : fraction_helix, "L": fraction_other }
            
            # DSSP assigns eight states: 310-helix (represented by G), alpha-helix (H), pi-helix (I), helix-turn (T), extended beta sheet (E), beta bridge (B), bend (S) and other/loop (L).

            frac_delta = []
            for specified_secstruc in args.dssp_fractions_specified.keys() :
                if args.dssp_fractions_specified[specified_secstruc] != None:
                    
                    curr_delta = args.dssp_fractions_specified[specified_secstruc] - measured_fractions[specified_secstruc] 
                    if curr_delta < 0.0 :
                        curr_delta = 0.0
                    frac_delta.append( np.abs( curr_delta )) 

            score = 0. + np.mean(frac_delta)  


        else:
            sys.exit("specified loss:  {loss_type}  \n not found in available losses!")

        scores.append(score)

    # Normalize loss weights vector.
    loss_weights_normalized = np.array(loss_weights) / np.sum(loss_weights)

    # Total loss for this neoantigen is the sum of its weighted scores.
    final_score = np.sum(np.array(scores) * loss_weights_normalized)
    loss = float(final_score)

    # The loss counts positively or negatively to the overall loss depending on whether this neoantigen is positively or negatively designed.
    # if neo.positive_design == True:
    #     loss = float(final_score)
    # else:
    #     loss = float(final_score) + 1

    return loss
