
# NASO (NeoAntigen Sequence Optimization)
A machine learning-based method to optimize the immunogenicity of HLA class I-restricted neoantigens.

## Quickstart

### 1. Install all requirements.

Using Docker: Download and run the docker image. Ensure you have [Docker](https://docs.docker.com/desktop/install/ubuntu/) installed.
```shell
docker pull lyrmagical/naso
docker run -it [--gpus all] --name naso lyrmagical/naso /bin/bash
git clone https://github.com/lyotvincent/Naso
```
[--gpus all] requires the installation of [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit), which allows users to run GPU accelerated containers.

### 2. Set up third-party libraries.
```shell
mv Third-party_libraries/* Naso/modules/
cd Naso/
```

### 3. Run Naso.
The raw data used in the paper are collected from [TSNAdb](http://biopharm.zju.edu.cn/tsnadb/), [NEPdb](http://nep.whu.edu.cn/) and [dbPepNeo](http://www.biostatistics.online/dbPepNeo/). The data files containing the neoantigens and their corresponding HLAs are in the `datasets` folder. Each line contains one raw neoantigen and the corresponding HLA, separated by a tab. More details about the raw data can be found in the paper.

Example:

```shell
python neoantigen_optimization.py --input datasets/NEPdb.txt --output outputs
```

## Outputs

A folder with the name specified by `--out` containing:
- Structures (`.pdb`) for each accepted move of the MCMC trajectory.
- A log file (`.out`) containing the scores at each step of the MCMC trajectory (accepted and rejected).

## Options
```
optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         the file containing the seed sequences. Each line contains one raw neoantigen and the corresponding HLA
                        (see examples in datasets folder). Must be specified.
  --output OUTPUT       the output folder. Must be specified.
  --mutation_rate MUTATION_RATE
                        number of mutations at each MCMC step (start-finish, stepped linear decay). Should probably be scaled
                        with neoantigen length (default: 3-1).
  --positions_fix POSITIONS_FIX
                        Whether to fix the positions where the mutation occurs. (default: True).
  --positions_mask POSITIONS_MASK
                        Masking positions where no mutation occurs. Choose from [None, default, custom]. None means no position
                        is masked, default means masking 1st, 2nd, and C-terminus amino acids, custom means masking the
                        corresponding HLA anchor residues (default: custom).
  --mutation_mask MUTATION_MASK
                        Whether to mask the self-defined positions, which should be added in the input file (in development).
                        (default: True).
  --select_positions SELECT_POSITIONS
                        how to select positions for mutation at each step. Choose from [random, plddt::quantile,
                        FILE.af2h::quantile]. TODO: plddt::quantile and FILE.af2h::quantile are in development. FILE.af2h needs
                        to be a file specifying the probability of mutation at each site. Optional arguments can be given with ::
                        e.g. plddt::0.25 will only mutate the 25% lowest plddt positions (default: random).
  --mutation_method MUTATION_METHOD
                        how to mutate selected positions. Choose from [uniform, frequency_adjusted, blosum62, pssm] (default:
                        uniform).
  --loss LOSS           the loss function used during optimization. Choose from [plddt, ptm, pae, bigmhc_el, bigmhc_im, entropy,
                        dual, tmalign (requires --template), dual_tmalign (requires --template), frac_dssp, min_frac_dssp
                        (requires --dssp_fractions_specified)]. TODO: tmalign, dual_tmalign, frac_dssp, min_frac_dssp are in
                        development. Multiple losses can be combined as a comma-separarted string of loss_name:args units (and
                        weighed with --loss_weights).
                        loss_0_name::loss0_param0;loss0_param1,loss_1_name::[loss_1_configfile.conf] ... (default:
                        dual,bigmhc_el,bigmhc_im).
  --loss_weights LOSS_WEIGHTS
                        if a combination of losses is passed, specify relative weights of each loss to the globabl loss by
                        providing a comma-separated list of relative weights. E.g. 2,1 will make the first loss count double
                        relative to the second one (default: 1,1,2).
  --T_init T_INIT       starting temperature for simulated annealing. Temperature is decayed exponentially (default: 0.01).
  --half_life HALF_LIFE
                        half-life for the temperature decay during simulated annealing (default: 500).
  --steps STEPS         number for steps for the MCMC trajectory (default: 300).
  --seed SEED           setting the seed (default: 42).
  --tolerance TOLERANCE
                        the tolerance on the loss sliding window for terminating the MCMC trajectory early (default: None).
  --model MODEL         AF2 model (_ptm) used during prediction. Choose from [1, 2, 3, 4, 5] (default: 3).
  --amber_relax AMBER_RELAX
                        amber relax pdbs written to disk, 0=do not relax, 1=relax every prediction (default: 1).
  --recycles RECYCLES   the number of recycles through the network used during structure prediction. Larger numbers increase
                        accuracy but linearly affect runtime (default: 3).
  --msa_clusters MSA_CLUSTERS
                        the number of MSA clusters used during feature generation (?). Larger numbers increase accuracy but
                        significantly affect runtime (default: 1).
  --template TEMPLATE   template PDB for use with tmalign-based losses (default: None).
  --dssp_fractions_specified DSSP_FRACTIONS_SPECIFIED
                        dssp fractions specfied for frac_dssp loss as E(beta sheet), H(alpha helix), notEH(other) e.g.
                        0.8,None,None will enforce 80% beta sheet; or 0.5,0,None will enforce 50% beta sheet, no helices
                        (default: None).
  --template_alignment TEMPLATE_ALIGNMENT
                        enforce tmalign alignment with fasta file (default: None).

```

## Acknowledgements

This work references the following separate libraries/datasets:

*   [oligomer_hallucination](https://github.com/bwicky/oligomer_hallucination)
*   [AlphaFold2](https://github.com/deepmind/alphafold)
*	[BigMHC](https://github.com/RosettaCommons/RoseTTAFold)
*	[TSNAdb](http://biopharm.zju.edu.cn/tsnadb/)
*	[NEPdb](http://nep.whu.edu.cn/)
*	[dbPepNeo](http://www.biostatistics.online/dbPepNeo/)

We thank all their contributors and maintainers!

## Contact

If you have any problems, just raise an issue in this repo.