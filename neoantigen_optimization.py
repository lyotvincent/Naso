import os, sys
import numpy as np
import copy

# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(script_dir + '/modules/') # import modules

from modules.arg_parser import *
from modules.mutations import *
from modules.af2 import *
from modules.losses import *

# Amino-acid frequencies taken from background frequencies of BLOSUM62.
# Data from https://www.ncbi.nlm.nih.gov/IEB/ToolBox/CPP_DOC/lxr/source/src/algo/blast/composition_adjustment/matrix_frequency_data.c
AA_freq = {'A': 0.07421620506799341,
           'R': 0.05161448614128464,
           'N': 0.044645808512757915,
           'D': 0.05362600083855441,
           'C': 0.02468745716794485,
           'Q': 0.03425965059141602,
           'E': 0.0543119256845875,
           'G': 0.074146941452645,
           'H': 0.026212984805266227,
           'I': 0.06791736761895376,
           'L': 0.09890786849715096,
           'K': 0.05815568230307968,
           'M': 0.02499019757964311,
           'F': 0.04741845974228475,
           'P': 0.038538003320306206,
           'S': 0.05722902947649442,
           'T': 0.05089136455028703,
           'W': 0.013029956129972148,
           'Y': 0.03228151231375858,
           'V': 0.07291909820561925}


class Neoantigen:

    def __init__(self, neo_sequence=None, paired_hla=None, mutation_index=None, position_weights=None):

        # Initialise sequences.
        self.init_sequence = neo_sequence
        self.hla = paired_hla

        if position_weights is None:
            self.position_weights = np.ones(len(neo_sequence)) / len(neo_sequence)
        else:
            self.position_weights = np.array(position_weights)
        
        # Initialise lengths.
        self.length = len(self.init_sequence)

        self.current_sequence = str(self.init_sequence)
        self.try_sequence = str(self.init_sequence)

        self.raw_mutation_index = mutation_index

        self.position_fix = False
        self.fix_flag = False

        self.position_mask = 'None'

        self.ref_mutable_positions = None
    
    # Method functions.
    def assign_mutable_positions(self, mutable_positions):
        '''Assign dictonary of neoantigens with arrays of mutable positions.'''
        self.mutable_positions = mutable_positions

    def assign_mutations(self, mutated_neoantigen):
        '''Assign mutated sequences to try_sequences.'''
        self.try_sequence = mutated_neoantigen

    def update_mutations(self):
        '''Update current sequences to try sequences.'''
        self.current_sequence = str(copy.deepcopy(self.try_sequence))
    
    def init_prediction(self, af2_prediction):
        '''Initalise scores/structure'''
        self.init_prediction_results, self.init_unrelaxed_structure = af2_prediction
        self.current_prediction_results = copy.deepcopy(self.init_prediction_results)
        self.current_unrelaxed_structure = copy.deepcopy(self.init_unrelaxed_structure)
        self.try_prediction_results = copy.deepcopy(self.init_prediction_results)
        self.try_unrelaxed_structure = copy.deepcopy(self.init_unrelaxed_structure)

    def init_loss(self, loss):
        '''Initalise loss'''
        self.init_loss = loss
        self.current_loss = float(self.init_loss)
        self.try_loss = float(self.init_loss)

    # def update_neo(self):
    #     '''Update current neoantigen sequence to try ones.'''
    #     self.current_sequence = str(self.try_sequence)

    def assign_prediction(self, af2_prediction):
        '''Assign try AlphaFold2 prediction (scores and structure).'''
        self.try_prediction_results, self.try_unrelaxed_structure = af2_prediction

    def update_prediction(self):
        '''Update current scores/structure to try scores/structure.'''
        self.current_unrelaxed_structure = copy.deepcopy(self.try_unrelaxed_structure)
        self.current_prediction_results = copy.deepcopy(self.try_prediction_results)

    def assign_loss(self, loss):
        '''Assign try loss.'''
        self.try_loss = float(loss)

    def update_loss(self):
        '''Update current loss to try loss.'''
        self.current_loss = float(self.try_loss)


def hallucinate_neoantigen(set_neoantigens, set_outputs, model_runners):
    outputs = set_outputs
    mutation_rate = '3-1'
    positions_select = 'random'
    positions_fix = True
    positions_mask = 'custom'
    mutation_mask = True
    seed = 42

    mutation_steps = 1000

    mutation_tolerance = None
    mutation_loss = [('dual', None), ('bigmhc_el', None), ('bigmhc_im', None)]
    loss_weights = [1.0, 1.0, 2.0]
    T_init = 0.01
    half_life = 500
    mutation_method = 'uniform'
    neo_weights = []
    to_amber_relax = 1

    os.makedirs(f'{outputs}_models', exist_ok=True)

    # Only one neoantigen can be processed at a time.
    assert len(set_neoantigens) == 1
    temp_pair = set_neoantigens[0]
    temp_pair[0] = temp_pair[0].upper()
    for o in temp_pair[0]:
        # Check if the neoantigen sequence is valid.
        if o not in list(AA_freq.keys()):
            print(f'Invalid neoantigen sequence:{temp_pair[0]}')
            return
    neoantigens = []
    if mutation_mask == True:
        neoantigen = Neoantigen(neo_sequence=temp_pair[0], paired_hla=temp_pair[1], mutation_index=temp_pair[2])
    else:
        neoantigen = Neoantigen(neo_sequence=temp_pair[0], paired_hla=temp_pair[1], mutation_index=None)
    neoantigens.append(neoantigen)
    if positions_fix == True:
        for i in range(len(neoantigens)):
            neoantigens[i].position_fix = True
    for i in range(len(neoantigens)):
        neoantigens[i].position_mask = positions_mask
    if to_amber_relax == 1:
        amber_relaxer = relax.AmberRelaxation(
            max_iterations=0,
            tolerance=2.39,
            stiffness=10.0,
            exclude_residues=[],
            max_outer_iterations=3,
            use_gpu=True)

    # MCMC WITH SIMULATED ANNEALING

    Mi, Mf = mutation_rate.split('-')
    M = np.linspace(int(Mi), int(Mf), mutation_steps) # stepped linear decay of the mutation rate

    current_loss = np.inf
    rolling_window = []
    rolling_window_width = 100
    for i in range(mutation_steps):

        if mutation_tolerance is not None and i > rolling_window_width: # check if change in loss falls under the tolerance threshold for terminating the simulation.

            if np.std(rolling_window[-rolling_window_width:]) < mutation_tolerance:
                print(f'The change in loss over the last 100 steps has fallen under the tolerance threshold ({mutation_tolerance}). Terminating the simulation...')
                sys.exit()
        else:

            # Update a few things.
            T = T_init * (np.exp(np.log(0.5) / half_life) ** i) # update temperature
            n_mutations = round(M[i]) # update mutation rate
            accepted = False # reset
            try_losses = []

            if i == 0: # do a first pass through the network before mutating anything -- baseline
                print('-' * 100)
                print('Starting...')
                for j in range(len(neoantigens)):
                    af2_prediction = predict_structure(neoantigens[j],
                                                       model_runners[j],
                                                       random_seed=np.random.randint(42))
                    neoantigens[j].init_prediction(af2_prediction) # assign
                    loss = compute_loss(mutation_loss,
                                        neoantigens[j],
                                        None,
                                        loss_weights) # calculate the loss
                    neoantigens[j].init_loss(loss) # assign
                    try_losses.append(loss) # increment global loss

            else:

                for j in range(len(neoantigens)):
                    # Mutate neoantigens sequences
                    if neoantigens[j].position_fix == True and i > 1:
                        neoantigens[j].fix_flag = True
                    neoantigens[j].assign_mutable_positions(select_positions(n_mutations, neoantigens[j], positions_select, None)) # define mutable positions for each neoantigen

                    neoantigens[j].assign_mutations(mutate(mutation_method, neoantigens[j], AA_freq)) # mutate those positions
                    
                    neoantigens[j].assign_prediction(predict_structure(neoantigens[j], model_runners[j], random_seed=seed))

                    loss = compute_loss(mutation_loss,
                                        neoantigens[j],
                                        None,
                                        loss_weights)
                    neoantigens[j].assign_loss(loss) # assign the loss to the object (for tracking)
                    try_losses.append(loss) # increment the global loss


            # Normalize neo weights vector.
            neo_weights = [1.0] * len(neoantigens)
            neo_weights_normalized = np.array(neo_weights) / np.sum(neo_weights)

            # Global loss is the weighted average of the individual neoantigen losses.
            try_loss = np.mean( np.array(try_losses) * neo_weights_normalized )

            delta = try_loss - current_loss # all losses must be defined such that optimising equates to minimising.

            # If the new solution is better, accept it.
            if delta < 0:
                accepted = True

                print(f'Step {i:05d}: change accepted >> LOSS {current_loss:2.3f} --> {try_loss:2.3f}')

                current_loss = float(try_loss) # accept loss change

                for j in range(len(neoantigens)):
                    print(f' > {j} loss  {neoantigens[j].current_loss:2.3f} --> {neoantigens[j].try_loss:2.3f}')
                    print(f' > {j} plddt {np.mean(neoantigens[j].current_prediction_results["plddt"]):2.3f} --> {np.mean(neoantigens[j].try_prediction_results["plddt"]):2.3f}')
                    print(f' > {j} ptm   {neoantigens[j].current_prediction_results["ptm"]:2.3f} --> {neoantigens[j].try_prediction_results["ptm"]:2.3f}')
                    print(f' > {j} pae   {np.mean(neoantigens[j].current_prediction_results["predicted_aligned_error"]):2.3f} --> {np.mean(neoantigens[j].try_prediction_results["predicted_aligned_error"]):2.3f}')
                    neoantigens[j].update_mutations() # accept sequence changes
                    neoantigens[j].update_prediction() # accept score/structure changes
                    neoantigens[j].update_loss() # accept loss change

                print('=' * 70)

            # If the new solution is not better, accept it with a probability of e^(-cost/temp).
            else:

                if np.random.uniform(0, 1) < np.exp( -delta / T):
                    accepted = True

                    print(f'Step {i:05d}: change accepted despite not improving the loss >> LOSS {current_loss:2.3f} --> {try_loss:2.3f}')

                    current_loss = float(try_loss)

                    for j in range(len(neoantigens)):
                        print(f' > {j} loss  {neoantigens[j].current_loss:2.3f} --> {neoantigens[j].try_loss:2.3f}')
                        print(f' > {j} plddt {np.mean(neoantigens[j].current_prediction_results["plddt"]):2.3f} --> {np.mean(neoantigens[j].try_prediction_results["plddt"]):2.3f}')
                        print(f' > {j} ptm   {neoantigens[j].current_prediction_results["ptm"]:2.3f} --> {neoantigens[j].try_prediction_results["ptm"]:2.3f}')
                        print(f' > {j} pae   {np.mean(neoantigens[j].current_prediction_results["predicted_aligned_error"]):2.3f} --> {np.mean(neoantigens[j].try_prediction_results["predicted_aligned_error"]):2.3f}')
                        neoantigens[j].update_mutations() # accept sequence changes
                        neoantigens[j].update_prediction() # accept score/structure changes
                        neoantigens[j].update_loss() # accept loss change

                    print('=' * 70)

                else:
                    accepted = False
                    print(f'Step {i:05d}: change rejected >> LOSS {current_loss:2.3f} !-> {try_loss:2.3f}')
                    print('-' * 70)

            sys.stdout.flush()

            # Save PDB if move was accepted.
            if accepted == True:

                for j in range(len(neoantigens)):

                    with open(f'{outputs}_models/{os.path.splitext(os.path.basename(outputs))[0]}_{j}_step_{str(i).zfill(5)}.pdb', 'w') as f:
                        # write pdb
                        if to_amber_relax == 0 :
                            f.write(protein.to_pdb(neoantigens[j].current_unrelaxed_structure))
                        elif to_amber_relax == 1 :
                            # to relax
                            relaxed_pdb, _, _ = amber_relaxer.process(prot=neoantigens[j].current_unrelaxed_structure)
                            f.write(relaxed_pdb)

            # Save scores for the step (even if rejected).
            # step accepted temperature mutations loss plddt ptm pae '
            score_string = f'{i:05d} '
            score_string += f'{accepted} '
            score_string += f'{T} '
            score_string += f'{n_mutations} '
            score_string += f'{try_loss} '
            score_string += f'{np.mean([np.mean(r.try_prediction_results["plddt"]) for r in neoantigens])} '
            score_string += f'{np.mean([r.try_prediction_results["ptm"] for r in neoantigens])} '
            score_string += f'{np.mean([np.mean(r.try_prediction_results["predicted_aligned_error"]) for r in neoantigens])}\t'

            for j in range(len(neoantigens)):
                score_string += f'{neoantigens[j].init_sequence},'
                score_string += f'{neoantigens[j].try_sequence},'
                score_string += f'{neoantigens[j].hla} '
                score_string += f'{neoantigens[j].try_loss} '
                score_string += f'{np.mean(neoantigens[j].try_prediction_results["plddt"])} '
                score_string += f'{neoantigens[j].try_prediction_results["ptm"]} '
                score_string += f'{np.mean(neoantigens[j].try_prediction_results["predicted_aligned_error"])}\t'

            with open(f'{outputs}_models/{os.path.splitext(os.path.basename(outputs))[0]}.out', 'a') as f:
                f.write(score_string + '\n')

            rolling_window.append(current_loss)

    print('Done')


if __name__ == '__main__':
    input_data = 'datasets/tsnadb_data.txt'
    raw_peptide_sequences = []
    with open(input_data, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            raw_peptide_sequences.append((line[0], line[1]))

    folder_index = 0

    # Setup AlphaFold2 models.
    model_id = 3
    model_recycles = 3
    model_msa_clusters = 1
    model_runners = setup_models(1, model_id=model_id, recycles=model_recycles, msa_clusters=model_msa_clusters)

    while raw_peptide_sequences != []:
        temp_peptide_sequences = []
        temp_peptide_sequences.append(raw_peptide_sequences[0])
        hallucinate_neoantigen(temp_peptide_sequences, f'outputs/{str(folder_index).zfill(5)}', model_runners)
        folder_index = folder_index + 1
        del raw_peptide_sequences[:1]
        np_raw_peptide_sequences = np.asarray(raw_peptide_sequences)
        np.savetxt(f'outputs/{str(folder_index).zfill(5)}_rest.txt', np_raw_peptide_sequences, fmt = '%s', delimiter = ',')
    
    print('all are processed')
    