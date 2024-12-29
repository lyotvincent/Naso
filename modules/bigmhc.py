import tempfile, os, subprocess
import matplotlib.pyplot as plt
import numpy as np
from Bio import pairwise2 as pw2
import argparse
import random
import torch
import csv
import sys

sys.path.append('bigmhc/src')
import cli, predict


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
        return False
    

def bigmhc_el_modelsetup():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.models = 'bigmhc/models/bat512' + ':' + 'bigmhc/models/bat1024' + ':' + 'bigmhc/models/bat2048' + ':' + 'bigmhc/models/bat4096' + ':' + 'bigmhc/models/bat8192' + ':' + 'bigmhc/models/bat16384' + ':' + 'bigmhc/models/bat32768'
    args.out = None
    args.pseudoseqs = 'bigmhc/data/pseudoseqs.csv'
    args.devices = 'all'
    args.verbose = False
    args.maxbat = 1
    args.hdrcnt = 0
    args.pepcol = 0
    args.tgtcol = None
    args.jobs = 1
    args.prefetch = 1
    args.saveatt = False
    args.train = False

    args = cli._parseTransferLearn(args)
    args = cli._parseModel(args)
    args = cli._parseDevices(args)
    args = cli._parseMaxbat(args)
    args = cli._parseOut(args)

    args = cli._parseHdrcnt(args)
    args = cli._parsePepcol(args)
    args = cli._parseTgtcol(args)

    args = cli._parseJobs(args)
    args = cli._parsePrefetch(args)

    models = cli._loadModels(args)

    args.modelname = "BigMHC_EL"

    return args, models


def bigmhc_im_modelsetup():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.models = 'bigmhc/models/bat512/im' + ':' + 'bigmhc/models/bat1024/im' + ':' + 'bigmhc/models/bat2048/im' + ':' + 'bigmhc/models/bat4096/im' + ':' + 'bigmhc/models/bat8192/im' + ':' + 'bigmhc/models/bat16384/im' + ':' + 'bigmhc/models/bat32768/im'
    args.out = None
    args.pseudoseqs = 'bigmhc/data/pseudoseqs.csv'
    args.devices = 'all'
    args.verbose = False
    args.maxbat = 1
    args.hdrcnt = 0
    args.pepcol = 0
    args.tgtcol = None
    args.jobs = 1
    args.prefetch = 1
    args.saveatt = False
    args.train = False

    args = cli._parseTransferLearn(args)
    args = cli._parseModel(args)
    args = cli._parseDevices(args)
    args = cli._parseMaxbat(args)
    args = cli._parseOut(args)

    args = cli._parseHdrcnt(args)
    args = cli._parsePepcol(args)
    args = cli._parseTgtcol(args)

    args = cli._parseJobs(args)
    args = cli._parsePrefetch(args)

    models = cli._loadModels(args)

    args.modelname = "BigMHC_IM"

    return args, models


def bigmhc_el_analyze(sequence_text, allele, args, models):
    pepfile = tempfile.mktemp()+'.csv'
    args.input = pepfile
    args.allele = allele
    args = cli._parseAllele(args)

    with open(pepfile ,'w') as f:
        writer = csv.writer(f)
        writer.writerow([sequence_text])
    f.close()
    data  = cli._loadData(args)

    preds = predict.predict(models=models, data=data, args=args)

    os.remove(pepfile)

    if preds.loc[0, 'mhc'] == allele and preds.loc[0, 'pep'] == sequence_text and is_number(preds.loc[0, 'BigMHC_EL']):
        el_score = float(preds.loc[0, 'BigMHC_EL'])
        return el_score
    print(f'Error in bigmhc_el_analyze() function:\n{preds}')
    sys.exit()


def bigmhc_im_analyze(sequence_text, allele, args, models):
    pepfile = tempfile.mktemp()+'.csv'
    args.input = pepfile
    args.allele = allele
    args = cli._parseAllele(args)

    with open(pepfile ,'w') as f:
        writer = csv.writer(f)
        writer.writerow([sequence_text])
    f.close()
    data  = cli._loadData(args)

    preds = predict.predict(models=models, data=data, args=args)

    os.remove(pepfile)

    if preds.loc[0, 'mhc'] == allele and preds.loc[0, 'pep'] == sequence_text and is_number(preds.loc[0, 'BigMHC_IM']):
        im_score = float(preds.loc[0, 'BigMHC_IM'])
        return im_score
    print(f'Error in bigmhc_im_analyze() function:\n{preds}')
    sys.exit()

