# %%
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

import pickle
import requests

import torch
from anndata import AnnData, read_h5ad
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
# %%

# sys.path.append("../")
sys.path.insert(0, "../")
#You may need to add scGPT to the python path
# export PYTHONPATH="${PYTHONPATH}:/app/mz_embed_engine/scgpt"

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt import prepare_data, prepare_dataloader, define_wandb_metrcis, evaluate, eval_testdata, train, test

from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# from misc import download_data_file
from sklearn.metrics import confusion_matrix


# %%

warnings.filterwarnings('ignore', category=UserWarning, message='^User provided device_type of \'cuda\', but CUDA is not available')

# create a class that takes a dictionary and creates class variables for each of the key and values in the input dictionary
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def show(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
              
# Download data
def download_data_file(dropbox_url, save_dir='data'):
    # Parse the file name from the URL
    file_name = dropbox_url.split("/")[-1]
    if '?' in file_name:
        file_name = file_name.split('?')[0]

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Send a GET request to the Dropbox URL
    response = requests.get(dropbox_url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the contents of the response to a file
        with open(os.path.join(save_dir, file_name), 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        print(f"Failed to download data from {dropbox_url}. Status code: {response.status_code}")

    return



hyperparameter_defaults = dict(
    # task="annotation",
    task = 'integration',
    seed=42,
    # dataset_name="metab_v0",
    dataset_name="metab_v1",
    do_train=True,
    # load_model="save/scGPT_bc",
    # load_model = None,
    mask_value=-1,
    pad_value=-2,
    include_zero_gene=True,
    pad_token="<pad>",
    mask_ratio=0.25, # ratio of masked values, default was 0.4
    epochs=10, #original was 30
    # n_bins=101, #counts/intensity bins, default was 51
    n_bins=51, #counts/intensity bins, default was 51
    GEP=True,  # (MLM) Gene expression prediction, Gene expression modelling
    GEPC=True,  #(MVC) Masked value prediction for cell embedding, Gene expression modelling for cell objective
    CLS=False,  # celltype classification objective
    # CLS =False,
    CCE =False,  # Contrastive cell embedding objective
    ESC=False,  # (ECS) Elastic similarity constraint, require ecs_thres>0
    ecs_thres=0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable. default was 0.8 in the paper it was 0.6
    DAR=False,  # (DAB) Domain adversarial loss
    DSBN = False, # Domain-spec batchnorm
    ADV = False,  # Adversarial training for batch correction
    input_style = "binned",  # "normed_raw", "log1p", or "binned"
    output_style = "binned",  # "normed_raw", "log1p", or "binned"
    input_emb_style = "continuous",  # "category" or "continuous" or "scaling"
    input_layer_key = "X_binned",
    dab_weight=0.0, # weight for domain adversarial loss
    cell_emb_style = "cls",  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs=0,  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs=0,  # delay epochs for domain adversarial loss
    lr=1e-4,
    batch_size=32, #default was 64
    layer_size=128,
    nlayers=4,
    nhead=4,
    # layer_size=64,
    # nlayers=2,
    # nhead=2,

    lr_ADV = 1e-3,  # learning rate for discriminator, used when ADV is True
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    schedule_interval=1, # interval of epochs for learning rate schedule
    save_eval_interval=2, #original was 5
    log_interval=100,
    fast_transformer=False, #need CUDA for this
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    n_hvg=False, # number of highly variable genes
    # n_hvg = 1200,  # Default number of highly variable genes
    n_hvp = 4000, # number of highly variable proteins
    # max_seq_len = 4001, # # Default n_hvg+1
    max_seq_len=1201, #1200 was the amount specified in the paper
    freeze = False, #freeze
    per_seq_batch_sample = False, # whether to sort samples by batch_id
    explicit_zero_prob = True, # whether explicit bernoulli for zeros
    normalize_total = False, # 3. whether to normalize the raw data and to what sum
    use_batch_labels = False, # whether to use batch labels, default was True
    use_mod = False, #modality aware? set to True for multi-omics
    do_sample_in_train = False, # sample the bernoulli in training, 

    # celltype_label="Cohort Label",
    celltype_label = 'Age Group',
    datasubset_label = 'pretrain_set',
    trainsubset_label = 'Train',
    valsubset_label = 'Val',
    testsubset_label = 'Test',

    load_model = None,
    # load_model = f"save/dev_metab_v0-May21-13-30", #10 epochs on Age Label
    # load_model = f"{data_dir}/save/dev_metab_v0-May16-14-47", #10 epochs on Cohort Label
    # load_model = f"{data_dir}/save/dev_metab_v0-May20-15-24", #30 epochs on IMDC Binary
    # celltype_label="IMDC Binary",
    # datasubset_label = 'finetune_set',
    # trainsubset_label = 'Finetune',
    # valsubset_label = 'Validation',
    # testsubset_label = 'Test',
    # celltype_label="sex",
    # datasubset_label = 'pretrain_set'

)         


def train_scgpt_wrapper(**kwargs):

    USE_WANDB = kwargs.get("USE_WANDB", True)

    data_dir = kwargs.get("data_dir", None)
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "DATA2")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    config_dict = {**hyperparameter_defaults}
    config_dict.update(kwargs)
    result_summary_dict = {}

    result_summary_dict.update(config_dict)
    start_time_str = time.strftime('%b%d-%H-%M')
    start_time = time.time()

    if USE_WANDB:
        print('using WandB')
        run = wandb.init(
            config=config_dict,
            project="scGPT",
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
        )
        config = wandb.config
        print(config)
        result_summary_dict.update({"wandb_run_id": run.id,
                                    'wandb_name': run.name,
                                    'start_time': start_time_str})
                                    # "wandb_url": run.url()})

        set_seed(config.seed)
    else:
        config = Config(config_dict)
        config.show()
        run = {}
        result_summary_dict.update({"wandb_run_id": None,
                                    'wandb_name': None,
                                    'start_time': start_time})
                                    # "wandb_url": None})

    set_seed(config.seed)



    # settings for input and preprocessing
    special_tokens = [config.pad_token, "<cls>", "<eoc>"]

    # %%
    dataset_name = config.dataset_name



    save_dir = Path(f"{data_dir}/save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    # os.system(f"cp {__file__} {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    # %%
    # ## Loading and preparing data
    if dataset_name == "metab_v1":
        print('this metabolomics data has already been log2 transformed, missing value filled and study-id standardized')
        data_url = 'https://www.dropbox.com/scl/fo/ltukt9smkbc60j88lkhn5/ADCoBKOkmF9L7LUKs7VdZbg?rlkey=pf19njmgm3mrex5a10y2qq4hf&dl=1'
        load_dir = Path(f"{data_dir}/{dataset_name}")
        load_dir.mkdir(parents=True, exist_ok=True)
        load_path = load_dir / "data.h5ad"
        if not os.path.exists(load_path):
            print('downloading data')
            download_data_file(data_url, save_dir=load_dir)
        adata = read_h5ad(load_path)

        if config.celltype_label == "IMDC Binary":
            adata = adata[adata.obs["IMDC"].isin(["FAVORABLE", "POOR"])].copy()
            adata.obs['IMDC Binary'] = adata.obs['IMDC']
    
        elif config.celltype_label == "MSKCC Binary":
            adata = adata[adata.obs["MSKCC"].isin(["FAVORABLE", "POOR"])].copy()
            adata.obs['MSKCC Binary'] = adata.obs['MSKCC']

        elif config.celltype_label == "Cohort Sex":
            cohort_vals = adata.obs['Cohort Label'].values
            sex_vals = adata.obs['sex'].values

            new_vals = [f'{x}_{y}' for x, y in zip(cohort_vals,sex_vals)]
            adata.obs['Cohort Sex'] = pd.Categorical(new_vals)

        elif config.celltype_label == "Age Group":
            adata.obs["Age Group"] = ['adult' if 'adult' in x else 'child' for x in adata.obs['Cohort Label']]

        elif config.celltype_label == "Dummy":
            adata.obs["Dummy"] = ['dummy' for x in adata.obs['Cohort Label']]

        # adata = scvi.data.pbmc_dataset()  # 11990 × 3346
        ori_batch_col = "Study ID"
        adata.obs["celltype"] = adata.obs[config.celltype_label].astype("category")
        
        # drop the nan celltype
        adata = adata[~adata.obs["celltype"].isna()]
        adata = adata[adata.obs["celltype"] != "nan"]
        
        data_is_raw = False
        filter_gene_by_counts = False
        adata_protein = None

    elif dataset_name == "metab_v0":
        # /Users/jonaheaton/Desktop/scGPT-main/data/metabolomics_apr24/data.h5ad
        data_url = 'https://www.dropbox.com/scl/fi/5g4ml5qio2ptumj8m3yjo/data.h5ad?rlkey=nlsrmacl5vzx9wxvci59vdj5p&dl=1'
        load_dir = Path(f"{data_dir}/{dataset_name}")
        load_dir.mkdir(parents=True, exist_ok=True)
        load_path = load_dir / "data.h5ad"
        if not os.path.exists(load_path):
            print('downloading data')
            download_data_file(data_url, save_dir=load_dir)
        adata = read_h5ad(load_path)

        print('Make sure that Study ID ST000422 is properly labeled as adult_other')
        # adata[adata.obs['Study ID'] == 'ST000422'].obs['Cohort Label'] = 'adult_other'
        print('All samples in ST001408 are male')
        # adata[adata.obs['Study ID'] == 'ST001408'].obs['sex'] = 'M'
        print('All samples in ST002023 are female')
        # adata[adata.obs['Study ID'] == 'ST002027'].obs['sex'] = 'F'

        adata.obs.loc[adata.obs['Study ID'] == 'ST000422', 'Cohort Label'] = 'adult_other'
        adata.obs.loc[adata.obs['Study ID'] == 'ST001408', 'sex'] = 'M'
        adata.obs.loc[adata.obs['Study ID'] == 'ST002027', 'sex'] = 'F'
        
        if config.celltype_label == "IMDC Binary":
            adata = adata[adata.obs["IMDC"].isin(["FAVORABLE", "POOR"])].copy()
            adata.obs['IMDC Binary'] = adata.obs['IMDC']
    
        elif config.celltype_label == "MSKCC Binary":
            adata = adata[adata.obs["MSKCC"].isin(["FAVORABLE", "POOR"])].copy()
            adata.obs['MSKCC Binary'] = adata.obs['MSKCC']

        elif config.celltype_label == "Cohort Sex":
            cohort_vals = adata.obs['Cohort Label'].values
            sex_vals = adata.obs['sex'].values

            new_vals = [f'{x}_{y}' for x, y in zip(cohort_vals,sex_vals)]
            adata.obs['Cohort Sex'] = pd.Categorical(new_vals)

        elif config.celltype_label == "Age Group":
            adata.obs["Age Group"] = ['adult' if 'adult' in x else 'child' for x in adata.obs['Cohort Label']]

        elif config.celltype_label == "Dummy":
            adata.obs["Dummy"] = ['dummy' for x in adata.obs['Cohort Label']]

        # adata = scvi.data.pbmc_dataset()  # 11990 × 3346
        ori_batch_col = "Study ID"
        adata.obs["celltype"] = adata.obs[config.celltype_label].astype("category")
        
        # drop the nan celltype
        adata = adata[~adata.obs["celltype"].isna()]
        adata = adata[adata.obs["celltype"] != "nan"]
        
        data_is_raw = True
        filter_gene_by_counts = False
        adata_protein = None

    if (config.use_mod) and (adata_protein is not None):
        gene_rna_df = pd.DataFrame(index = adata.var.index.tolist())
        gene_rna_df['mod'] = 'RNA'
        gene_protein_df = pd.DataFrame(index = adata_protein.var.index.tolist())
        gene_protein_df['mod'] = 'Protein'
        gene_loc_df = pd.concat([gene_rna_df, gene_protein_df])
        gene_loc_df['mod'] = gene_loc_df['mod'].astype('category')

    elif (config.use_mod) and (adata_protein is None):
        raise ValueError('adata_protein is None')


    # %%
    # make the batch category column
    adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    num_batch_types = len(set(batch_id_labels))
    print(f"number of batch types: {num_batch_types}")
    adata.obs["batch_id"] = batch_id_labels

    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    print(f"number of cell types: {num_types}")
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels


    adata.var["gene_name"] = adata.var.index.tolist()
    # %%

    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # model
        if os.path.exists(model_config_file):
            with open(model_config_file, "r") as f:
                model_configs = json.load(f)
            logger.info(
                f"Resume model from {model_file}, the model args will be overriden by the "
                f"config {model_config_file}."
            )
        else:
            model_configs = {}
            model_configs["embsize"] = config.layer_size
            model_configs["nheads"] = config.nhead
            model_configs["d_hid"] = config.layer_size
            model_configs["nlayers"] = config.nlayers

        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]

        backup_model_config_file = save_dir / "args.json"
        with open(backup_model_config_file, "w") as f:
            # json.dump(config.__dict__, f, indent=2)
            # json.dump(backup_model_config_file, f, indent=2)
            json.dump(model_configs, f, indent=2)
    else:
        embsize = config.layer_size 
        nhead = config.nhead
        nlayers = config.nlayers  
        d_hid = config.layer_size
        vocab_file = save_dir / "vocab.json"
        model_config_file = save_dir / "args.json"
        # save the config to json

        model_configs = {}
        model_configs["embsize"] = config.layer_size
        model_configs["nheads"] = config.nhead
        model_configs["d_hid"] = config.layer_size
        model_configs["nlayers"] = config.nlayers
        with open(model_config_file, "w") as f:
            # json.dump(config.__dict__, f, indent=2)
            json.dump(model_configs, f, indent=2)


    # %%
    # set up the preprocessor, use the args to config the workflow
    print('Preprocess data')
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=config.normalize_total,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=config.n_hvg,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)

    # %%
    # This sorting caused crashed in earlier attempts, need to check if it still crashes
    if config.per_seq_batch_sample:
        # sort the adata by batch_id in advance
        adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

    # %% [markdown]
    # ## Tokenize input
    print('Tokenize input')

    # %%
    input_layer_key = config.input_layer_key
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    datasubset_label = config.datasubset_label

    if datasubset_label in adata.obs:
        trainsubset_label = config.trainsubset_label
        valsubset_label = config.valsubset_label
        testsubset_label = config.testsubset_label

        print(f'use {datasubset_label} for train ({trainsubset_label}) /val ({valsubset_label}) split')
        train_data = all_counts[adata.obs[datasubset_label] == trainsubset_label]
        valid_data = all_counts[adata.obs[datasubset_label] == valsubset_label]
        train_celltype_labels = celltypes_labels[adata.obs[datasubset_label] == trainsubset_label]
        valid_celltype_labels = celltypes_labels[adata.obs[datasubset_label] == valsubset_label]
        train_batch_labels = batch_ids[adata.obs[datasubset_label] == trainsubset_label]
        valid_batch_labels = batch_ids[adata.obs[datasubset_label] == valsubset_label]
        print('train/val split: ', len(train_data), len(valid_data))

        adata_train = adata[adata.obs[datasubset_label] == trainsubset_label].copy()
        adata_test = adata[adata.obs[datasubset_label] == testsubset_label].copy()

    else:
        print('use random split for train/val split')
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
        )

        adata_train = adata.copy()
        adata_test = None
    # %%
    if config.load_model is None:
        vocab = GeneVocab(
            genes,
            special_tokens,
        )
        vocab.save_json(vocab_file)
    else:
        vocab.save_json(save_dir / "vocab.json")
        # vocab = Vocab(
        #     VocabPybind(genes + special_tokens, None)
        # )  # bidirectional lookup [gene <-> int]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    # %%
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=config.include_zero_gene,
        # mod_type=mod_type if config.use_mod else None,
        # vocab_mod=vocab_mod if config.use_mod else None,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=config.include_zero_gene,
        # mod_type=mod_type if config.use_mod else None,
        # vocab_mod=vocab_mod if config.use_mod else None,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )




    # %% [markdown]
    # # Create and finetune scGPT

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    print(f"vocab size: {ntokens}")
    model = TransformerModel(
        ntoken=ntokens,
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        # nlayers_cls: int = 3,
        n_cls=num_types,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.GEPC,
        do_dab=config.DAR,
        use_batch_labels=config.use_batch_labels,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        # input_emb_style: str = "continuous",
        n_input_bins=config.n_bins,
        # cell_emb_style: str = "cls",
        # mvc_decoder_style: str = "inner product",
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
        # use_mod=config.use_mod,
        # ntokens_mod=ntokens_mod if config.use_mod else None,
        # vocab_mod=vocab_mod if config.use_mod else None,
    )


    if config.load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)
    if USE_WANDB:
        wandb.watch(model)


    criterion_gep_gepc = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=config.schedule_interval, 
                                                gamma=config.schedule_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    if config.ADV:
        discriminator = AdversarialDiscriminator(
            d_model=embsize,
            n_cls=num_batch_types,
        ).to(device)

        criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
        optimizer_E = torch.optim.Adam(model.parameters(), lr=config.lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, 
            step_size=config.schedule_interval, 
            gamma=config.schedule_ratio
        )
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, 
            step_size=config.schedule_interval,
            gamma=config.schedule_ratio
        )

    # %%
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None

    if USE_WANDB:
        define_wandb_metrcis()

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(
            tokenized_train= tokenized_train,
            tokenized_valid= tokenized_valid,
            train_batch_labels=train_batch_labels,
            valid_batch_labels=valid_batch_labels,
            config = config,
            epoch = epoch,
            train_celltype_labels=train_celltype_labels,
            valid_celltype_labels=valid_celltype_labels,
            sort_seq_batch=config.per_seq_batch_sample)
        
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
            per_seq_batch_sample=config.per_seq_batch_sample,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
            per_seq_batch_sample=config.per_seq_batch_sample,
        )

        if config.do_train:
            train(
                model = model,
                loader=train_loader,
                vocab=vocab,
                criterion_gep_gepc=criterion_gep_gepc if config.GEP or config.GEPC else None,
                # criterion_gep_gepc=criterion_gep_gepc if config.GEP and config.GEPC else None,
                criterion_dab=criterion_dab if config.DAR else None,
                criterion_cls=criterion_cls if config.CLS else None,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=config,
                logger=logger,
                epoch=epoch,
                use_wandb=USE_WANDB,
            )
        # val_loss, val_mre = evaluate(
        val_loss = evaluate(
            model,
            loader=valid_loader,
            vocab=vocab,
            criterion_gep_gepc=criterion_gep_gepc if config.GEP or config.GEPC else None,
            # criterion_gep_gepc=criterion_gep_gepc if config.GEP and config.GEPC else None,
            criterion_dab=criterion_dab if config.DAR else None,
            criterion_cls=criterion_cls if config.CLS else None,
            device=device,
            config=config,
            epoch=epoch
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            # f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
            f"valid loss/mse {val_loss:5.4f} |"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")
            metrics_to_log = {"epoch": epoch, "val/loss": val_loss}

            # eval on traindata
            results = eval_testdata(
                best_model,
                adata_t = adata_sorted if config.per_seq_batch_sample else adata,
                gene_ids=gene_ids,
                vocab=vocab,
                config=config,
                logger=logger,
                include_types=["cls"],
                subset_label = config.trainsubset_label,
                criterion_gep_gepc=criterion_gep_gepc if config.GEP or config.GEPC else None,
                # criterion_gep_gepc=criterion_gep_gepc if config.GEP and config.GEPC else None,
                criterion_dab=criterion_dab if config.DAR else None,
                criterion_cls=criterion_cls if config.CLS else None,
                device=device,
                # epoch=epoch
            )
            results["batch_umap"].savefig(
                save_dir / f"train_embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300, bbox_inches="tight"
            )

            results["celltype_umap"].savefig(
                save_dir / f"train_embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300, bbox_inches="tight"
            )
            metrics_to_log.update({"train/" + k: v for k, v in results.items()})


            # eval on val data
            results = eval_testdata(
                best_model,
                adata_t = adata_sorted if config.per_seq_batch_sample else adata,
                # adata_t=adata,
                gene_ids=gene_ids,
                vocab=vocab,
                config=config,
                logger=logger,
                include_types=["cls"],
                subset_label = config.valsubset_label,
                criterion_gep_gepc=criterion_gep_gepc if config.GEP or config.GEPC else None,
                # criterion_gep_gepc=criterion_gep_gepc if config.GEP and config.GEPC else None,
                criterion_dab=criterion_dab if config.DAR else None,
                criterion_cls=criterion_cls if config.CLS else None,
                device=device,
                # epoch=epoch
            )
            results["batch_umap"].savefig(
                save_dir / f"val_embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300, bbox_inches="tight"
            )

            results["celltype_umap"].savefig(
                save_dir / f"val_embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300, bbox_inches="tight"
            )
            # metrics_to_log = {"val/" + k: v for k, v in results.items()}
            metrics_to_log.update({"val/" + k: v for k, v in results.items()})

            # eval on testdata
            results = eval_testdata(
                best_model,
                adata_t = adata_sorted if config.per_seq_batch_sample else adata,
                # adata_t=adata,
                gene_ids=gene_ids,
                vocab=vocab,
                config=config,
                logger=logger,
                include_types=["cls"],
                subset_label = config.testsubset_label,
                criterion_gep_gepc=criterion_gep_gepc if config.GEP or config.GEPC else None,
                # criterion_gep_gepc=criterion_gep_gepc if config.GEP and config.GEPC else None,
                criterion_dab=criterion_dab if config.DAR else None,
                criterion_cls=criterion_cls if config.CLS else None,
                device=device,
                # epoch=epoch
            )

            results["batch_umap"].savefig(
                save_dir / f"test_embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300, bbox_inches="tight"
            )

            results["celltype_umap"].savefig(
                save_dir / f"test_embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300, bbox_inches="tight"
            )
            # metrics_to_log = {"test/" + k: v for k, v in results.items()}
            metrics_to_log.update({"test/" + k: v for k, v in results.items()})
            
            if USE_WANDB:
                metrics_to_log["train/batch_umap"] = wandb.Image(
                    str(save_dir / f"train_embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )

                metrics_to_log["train/celltype_umap"] = wandb.Image(
                    str(save_dir / f"train_embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )

                metrics_to_log["val/batch_umap"] = wandb.Image(
                    str(save_dir / f"val_embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )

                metrics_to_log["val/celltype_umap"] = wandb.Image(
                    str(save_dir / f"val_embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )
                
                metrics_to_log["test/batch_umap"] = wandb.Image(
                    str(save_dir / f"test_embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )

                metrics_to_log["test/celltype_umap"] = wandb.Image(
                    str(save_dir / f"test_embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
                    caption=f"celltype avg_bio epoch {best_model_epoch}",
                )
                metrics_to_log["test/best_model_epoch"] = best_model_epoch
                wandb.log(metrics_to_log)
                wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

        scheduler.step()
        if config.ADV:
            scheduler_D.step()
            scheduler_E.step()

    # %%
    # save the best model
    result_summary_dict.update(metrics_to_log)
    if config.epochs > 0:
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    else:
        best_model = model
        torch.save(best_model.state_dict(), save_dir / "best_model.pt")

    # %% [markdown]
    # ## Gene embeddings


    # %% Annotation of CellType Results
    if (config.task == "annotation") and (adata_test is not None):

        predictions, labels, results, cell_embeddings = test(model=best_model,
                                            adata=adata_test,
                                            gene_ids=gene_ids,
                                            vocab=vocab,
                                            config=config,
                                            device=device,
                                            logger=logger)
        # adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]
        adata_test.obs["predictions"] = [id2type[p] for p in predictions]
        adata_test.obsm["X_scGPT"] = cell_embeddings

        # plot
        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] 
        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}

        # Compute the UMAP if it doesn't exist
        if 'X_umap' not in adata_test.obsm.keys():
            # sc.pp.neighbors(adata_test)
            # sc.tl.umap(adata_test)
            sc.pp.neighbors(adata_test, use_rep="X_scGPT")
            sc.tl.umap(adata_test, min_dist=0.3)


        with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (300)}):
            sc.pl.umap(
                adata_test,
                # adata_test_raw,
                color=["celltype", "predictions"],
                palette=palette_,
                show=False,
            )
            plt.savefig(save_dir / "results.png", dpi=300)

        save_dict = {
            "predictions": predictions,
            "labels": labels,
            "results": results,
            "id_maps": id2type
        }
        with open(save_dir / "results.pkl", "wb") as f:
            pickle.dump(save_dict, f)

        results["test/cell_umap"] = wandb.Image(
            str(save_dir / "results.png"),
            caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
        )
        wandb.log(results)


        # %%

        celltypes = list(celltypes)
        for i in set([id2type[p] for p in predictions]):
            if i not in celltypes:
                celltypes.remove(i)
        cm = confusion_matrix(labels, predictions)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
        plt.savefig(save_dir / "confusion_matrix.png", dpi=300)

        results["test/confusion_matrix"] = wandb.Image(
            str(save_dir / "confusion_matrix.png"),
            caption=f"confusion matrix",
        )

        result_summary_dict.update(results)

    print('results saved in: ')
    print(save_dir)

    # %%
    if USE_WANDB:
        artifact = wandb.Artifact(f"best_model", type="model")
        glob_str = os.path.join(save_dir, "best_model.pt")
        artifact.add_file(glob_str)
        run.log_artifact(artifact)

        run.finish()
        wandb.finish()
    gc.collect()

    result_summary_dict['save_dir'] = save_dir
    # record the end time and the total elapsed time
    end_time_str = time.strftime('%b%d-%H-%M')
    result_summary_dict.update({"end_time": end_time_str})
    result_summary_dict.update({"total_time": time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))})


    return result_summary_dict


def clean_res_values(k,v):
    # check if the value is an image
    if isinstance(v,wandb.Image):
        return False
    elif isinstance(v,plt.Figure):
        return False
    return True

# %%
if __name__ == "__main__":
    # train_scgpt_wrapper()
    results_dict = train_scgpt_wrapper(epochs=3,layer_size=64,nlayers=2,nhead=2)
    # train_scgpt_wrapper(epochs=1, load_model = f"{data_dir}/save/dev_metab_v0-May20-15-24")
    # train_scgpt_wrapper(epochs=1, load_model = f"{data_dir}/save/dev_metab_v0-May21-13-30")
    # train_scgpt_wrapper(epochs=1, load_model = f"{data_dir}/save/dev_metab_v0-May21-13-30", celltype_label="IMDC Binary", datasubset_label = 'finetune_set', trainsubset_label = 'Finetune', valsubset_label = 'Validation', testsubset_label = 'Test')


# %%
# save results dict to json
    print(results_dict)
    os.makedirs('results', exist_ok=True)

    # remove the images from the results dict
    results_dict = {k: v for k, v in results_dict.items() 
                           if clean_res_values(k,v)}
    with open('results/results_dict.json', 'w') as fp:
        json.dump(results_dict, fp)
