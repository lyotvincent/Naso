# arg parser module

import argparse
import sys, datetime
import numpy as np

def get_args():
    ''' Parse input arguments'''

    parser = argparse.ArgumentParser(
            description='NASO (NeoAntigen Sequence Optimization)'
            )

    parser.add_argument(
            '--input',
            help='the file containing the seed sequences. Each line contains one raw neoantigen and the corresponding HLA (see examples in datasets folder). Must be specified.',
            action='store',
            type=str,
            required=True
            )

    parser.add_argument(
            '--output',
            help='the output folder. Must be specified.',
            action='store',
            type=str,
            required=True
            )

    parser.add_argument(
            '--mutation_rate',
            default='3-1',
            action='store',
            help='number of mutations at each MCMC step (start-finish, stepped linear decay). Should probably be scaled with neoantigen length (default: %(default)s).'
            )
    
    parser.add_argument(
            '--positions_fix',
            default=True,
            action='store',
            type=bool,
            help='Whether to fix the positions where the mutation occurs. (default: %(default)s).'
            )
    
    parser.add_argument(
            '--positions_mask',
            default='custom',
            action='store',
            type=str,
            help='Masking positions where no mutation occurs. Choose from [None, default, custom].\
            None means no position is masked, default means masking 1st, 2nd, and C-terminus amino acids, custom means masking the corresponding HLA anchor residues (default: %(default)s).'
            )
    
    parser.add_argument(
            '--mutation_mask',
            default=True,
            action='store',
            type=bool,
            help='Whether to mask the self-defined positions, which should be added in the input file (in development). (default: %(default)s).'
            )

    parser.add_argument(
            '--select_positions',
            default='random',
            action='store',
            type=str,
            help='how to select positions for mutation at each step. Choose from [random, plddt::quantile, FILE.af2h::quantile].\
            TODO: plddt::quantile and FILE.af2h::quantile are in development. FILE.af2h needs to be a file specifying the probability of mutation at each site.\
            Optional arguments can be given with :: e.g. plddt::0.25 will only mutate the 25%% lowest plddt positions (default: %(default)s).'
            )

    parser.add_argument(
            '--mutation_method',
            default='uniform',
            action='store',
            type=str,
            help='how to mutate selected positions. Choose from [uniform, frequency_adjusted, blosum62, pssm] (default: %(default)s).'
            )

    parser.add_argument(
            '--loss',
            default='dual,bigmhc_el,bigmhc_im',
            type=str,
            help='the loss function used during optimization. Choose from \
            [plddt, ptm, pae, bigmhc_el, bigmhc_im, entropy, dual, tmalign (requires --template), dual_tmalign (requires --template), \
            frac_dssp, min_frac_dssp (requires --dssp_fractions_specified)].\
            TODO: tmalign, dual_tmalign, frac_dssp, min_frac_dssp are in development. Multiple losses can be combined as a comma-separarted string of loss_name:args units (and weighed with --loss_weights).\
            loss_0_name::loss0_param0;loss0_param1,loss_1_name::[loss_1_configfile.conf] ... \
             (default: %(default)s).'
            )

    parser.add_argument(
            '--loss_weights',
            default='1,1,2',
            type=str,
            action='store',
            help='if a combination of losses is passed, specify relative weights of each loss to the globabl loss by providing a comma-separated list of relative weights. \
            E.g. 2,1 will make the first loss count double relative to the second one (default: %(default)s).'
            )

    parser.add_argument(
            '--T_init',
            default=0.01,
            action='store',
            type=float,
            help='starting temperature for simulated annealing. Temperature is decayed exponentially (default: %(default)s).'
            )

    parser.add_argument(
            '--half_life',
            default=500,
            action='store',
            type=float,
            help='half-life for the temperature decay during simulated annealing (default: %(default)s).'
            )

    parser.add_argument(
            '--steps',
            default=300,
            action='store',
            type=int,
            help='number for steps for the MCMC trajectory (default: %(default)s).'
            )
    
    parser.add_argument(
            '--seed',
            default=42,
            action='store',
            type=int,
            help='setting the seed (default: %(default)s).'
            )

    parser.add_argument(
            '--tolerance',
            default=None,
            action='store',
            type=float,
            help='the tolerance on the loss sliding window for terminating the MCMC trajectory early (default: %(default)s).'
            )

    parser.add_argument(
            '--model',
            default=3,
            action='store',
            type=int,
            help='AF2 model (_ptm) used during prediction. Choose from [1, 2, 3, 4, 5] (default: %(default)s).'
            )

    parser.add_argument(
            '--amber_relax',
            default=1,
            action='store',
            type=int,
            help='amber relax pdbs written to disk, 0=do not relax, 1=relax every prediction (default: %(default)s).'
            )

    parser.add_argument(
            '--recycles',
            default=3,
            action='store',
            type=int,
            help='the number of recycles through the network used during structure prediction. Larger numbers increase accuracy but linearly affect runtime (default: %(default)s).'
            )

    parser.add_argument(
            '--msa_clusters',
            default=1,
            action='store',
            type=int,
            help='the number of MSA clusters used during feature generation (?). Larger numbers increase accuracy but significantly affect runtime (default: %(default)s).'
            )

    parser.add_argument(
            '--template',
            default=None,
            type=str,
            action='store',
            help='template PDB for use with tmalign-based losses (default: %(default)s).'
            )

    parser.add_argument(
            '--dssp_fractions_specified',
            default=None,
            type=str,
            action='store',
            help='dssp fractions specfied for frac_dssp loss as E(beta sheet), H(alpha helix), notEH(other)\
            e.g. 0.8,None,None will enforce 80%% beta sheet; or 0.5,0,None will enforce 50%% beta sheet, no helices (default: %(default)s).'
            )

    parser.add_argument(
            '--template_alignment',
            default=None,
            type=str,
            action='store',
            help='enforce tmalign alignment with fasta file (default: %(default)s).'
            )

    args = parser.parse_args()

    # LOSSSES
    # all losses stored in list of tuples [ ( loss_name, [loss_param0, loss_param1] ), ...]
    # losses processed in order
    # if no parameters necessary list entry is empty list/None
    losses = []
    for curr_loss_str in args.loss.strip(',').split(','):
        loss_parameters = []

        if '::' in curr_loss_str:
            loss_name, loss_arguments = curr_loss_str.split('::')

            for curr_loss_param in loss_arguments.strip(';').split(';'):
                if "[" in curr_loss_param:
                    print("loss configfile NOT IMPLEMENTED YET. System exiting...")
                    sys.exit()
                else:
                    #assuming all parameters are number
                    loss_parameters.append( float(curr_loss_param) )
        else:
            loss_name = str(curr_loss_str)
            loss_parameters = None

        losses.append((loss_name, loss_parameters))

    # replace args.loss string with new dictionary
    args.loss = losses

    # loss weights only relevant if more than one loss declared
    if args.loss_weights != None:
        # split relative weight for losses from input string,
        # update as list
        loss_weights = []
        for curr_loss_weight in args.loss_weights.strip(',').split(','):
            if "-" in curr_loss_weight :
                # loss is ramped over course of run
                print("loss weight ramping NOT IMPLMENTED YET. System exiting...")
                sys.exit()
                loss_weights.append( [ float(i) for i in curr_loss_weight.split("-") ] )
                pass
            else:
                #loss is constant over course of run (ramped to idential values)
                # loss_weights.append( [ float(curr_loss), float(curr_loss) ] )
                loss_weights.append( float(curr_loss_weight) )

        args.loss_weights = loss_weights

        assert len(args.loss_weights) == len(args.loss)
    else:
        # no loss weights declared, all losses weighed equally
        args.loss_weights = [1.0] * len(args.loss) # processed list of losses

    args.position_weights = None

    # UPDATE / MUTATIONS
    # reading in additional arguments for updates, currently just option for
    # quantile to mutate when plddt or .af2h specified
    args.select_position_params = None
    if '::' in args.select_positions:
        # additional arguments in args.update
        args.select_positions, select_position_params = args.select_positions.split('::')
        args.select_position_params = float(select_position_params)
        print (" Select position params set to ", args.select_position_params)

    return args