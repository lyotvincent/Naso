import os, sys

# AlphaFold2 new release (May 3, 2023): commit 6a3af1adb3bbbc53562da100e3819b2fc882f915
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir + '/alphafold')

from alphafold.data import pipeline
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.relax import relax

import mock
import numpy as np
import time


def setup_models(neo_num, model_id=3, recycles=3, msa_clusters=1):
    '''Setup AlphaFold2 models.'''

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    model_runners = []
    model_name = f'model_{model_id}_ptm'
    model_config = config.model_config(model_name)
    model_config.model.global_config.subbatch_size = 32
    model_config.data.common.num_recycle = recycles # AF2 default is 3. Effect on computing time is linear.
    model_config.model.num_recycle = recycles # AF2 default is 3. Effect on computing time is linear.
    model_config.data.common.max_extra_msa = msa_clusters  # AF2 default is 5120. Turning off is about 8x faster.
    model_config.data.eval.max_msa_clusters = msa_clusters # AF2 default is 512. Turning off is about 8x faster.
    model_config.data.eval.num_ensemble = 1
    
    model_params = data.get_model_haiku_params(model_name=model_name, data_dir='af2_models')
    model_runner = model.RunModel(model_config, model_params)


    for i in range(neo_num):
        model_runners.append(model_runner)

    return model_runners


def mk_mock_template(query_sequence):
    '''Generate mock template features from the input sequence.'''

    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in query_sequence:
        templates_all_atom_positions.append(np.zeros((templates.residue_constants.atom_type_num, 3)))
        templates_all_atom_masks.append(np.zeros(templates.residue_constants.atom_type_num))
        output_templates_sequence.append('-')
        output_confidence_scores.append(-1)

    output_templates_sequence = ''.join(output_templates_sequence)
    templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID)

    template_features = {'template_all_atom_positions': np.array(templates_all_atom_positions)[None],
                         'template_all_atom_masks': np.array(templates_all_atom_masks)[None],
                         'template_sequence': [f'none'.encode()],'template_aatype': np.array(templates_aatype)[None],
                         'template_confidence_scores': np.array(output_confidence_scores)[None],
                         'template_domain_names': [f'none'.encode()],
                         'template_release_date': [f'none'.encode()]
                        }

    return template_features


def predict_structure(neo_object,
                      model_runner: model.RunModel,
                      random_seed=0):
    query_sequence = neo_object.try_sequence

    parsers_Msa = parsers.Msa(sequences=[query_sequence],
                              deletion_matrix=[[0]*len(query_sequence)],
                              descriptions=['none'])

    data_pipeline_mock = mock.Mock()
    data_pipeline_mock.process.return_value = {
        **pipeline.make_sequence_features(sequence=query_sequence,
                                          description="none",
                                          num_res=len(query_sequence)),
        **pipeline.make_msa_features(msas=[parsers_Msa]),
        **mk_mock_template(query_sequence)
    }

    start = time.time()
    feature_dict = data_pipeline_mock.process()
    processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
    prediction_result = model_runner.predict(processed_feature_dict, random_seed=random_seed)

    plddt = prediction_result['plddt']
    plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=True)
    prediction_result['plddt'] = prediction_result['plddt'] / 100.0

    end = time.time()
    print(f'({query_sequence} prediction took {(end - start):.1f} s)')

    return prediction_result, unrelaxed_protein


if __name__ == '__main__':
    model_runners = setup_models(1, model_id=3, recycles=3, msa_clusters=1)