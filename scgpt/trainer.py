import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
import traceback
import numpy as np

from anndata import AnnData
import scanpy as sc
from typing import List, Tuple, Dict, Optional
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import SubsetsBatchSampler
from scgpt.loss import (
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.utils import eval_scib_metrics
import warnings
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def prepare_dataset(
    tokenized_data,
    batch_labels,
    config,
    celltype_labels=None,
    sort_seq_batch=False,
    epoch=0,
) -> Dict[str, torch.Tensor]:
    
    masked_values = random_mask_value(
        tokenized_data["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )

    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values == config.mask_value).sum() / (masked_values - config.pad_value).count_nonzero():.4f}",
    )

    input_gene_ids = tokenized_data["genes"]
    input_values = masked_values
    target_values = tokenized_data["values"]

    tensor_batch_labels = torch.from_numpy(batch_labels).long()

    data_pt = {
        "gene_ids": input_gene_ids,
        "values": input_values,
        "target_values": target_values,
        "batch_labels": tensor_batch_labels,
    }

    if config.task == "annotation":
        tensor_celltype_labels = torch.from_numpy(celltype_labels).long()
        data_pt["celltype_labels"] = tensor_celltype_labels

    if config.task == "multiomic":
        tensor_mod_types = tokenized_data["mod_types"].long()
        data_pt["mod_types"] = tensor_mod_types

    if sort_seq_batch:
        sort_ids = np.argsort(batch_labels)
        for key in data_pt.keys():
            data_pt[key] = data_pt[key][sort_ids]

    return data_pt


def prepare_data(
    tokenized_train,
    tokenized_valid,
    train_batch_labels,
    valid_batch_labels,
    config,
    epoch,
    train_celltype_labels=None,
    valid_celltype_labels=None,
    sort_seq_batch=False,
) -> Tuple[Dict[str, torch.Tensor]]:
    assert config.task in ["annotation", "integration", "perturb", "multiomic"]


    train_data_pt = prepare_dataset(
        tokenized_train,
        train_batch_labels,
        config,
        celltype_labels=train_celltype_labels,
        sort_seq_batch=sort_seq_batch,
        epoch=epoch,
    )

    valid_data_pt = prepare_dataset(
        tokenized_valid,
        valid_batch_labels,
        config,
        celltype_labels=valid_celltype_labels,
        sort_seq_batch=sort_seq_batch,
        epoch=epoch,
    )

    return train_data_pt, valid_data_pt


# def prepare_data(
#     tokenized_train,
#     tokenized_valid,
#     train_batch_labels,
#     valid_batch_labels,
#     config,
#     epoch,
#     train_celltype_labels=None,
#     valid_celltype_labels=None,
#     sort_seq_batch=False,
# ) -> Tuple[Dict[str, torch.Tensor]]:
#     assert config.task in ["annotation", "integration", "perturb", "multiomic"]
#     masked_values_train = random_mask_value(
#         tokenized_train["values"],
#         mask_ratio=config.mask_ratio,
#         mask_value=config.mask_value,
#         pad_value=config.pad_value,
#     )
#     masked_values_valid = random_mask_value(
#         tokenized_valid["values"],
#         mask_ratio=config.mask_ratio,
#         mask_value=config.mask_value,
#         pad_value=config.pad_value,
#     )
#     print(
#         f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
#         f"{(masked_values_train == config.mask_value).sum() / (masked_values_train - config.pad_value).count_nonzero():.4f}",
#     )

#     input_gene_ids_train, input_gene_ids_valid = (
#         tokenized_train["genes"],
#         tokenized_valid["genes"],
#     )
#     input_values_train, input_values_valid = masked_values_train, masked_values_valid

#     target_values_train, target_values_valid = (
#         tokenized_train["values"],
#         tokenized_valid["values"],
#     )

#     tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
#     tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

#     if config.task == "annotation":
#         tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
#         tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

#     if config.task == "multiomic":
#         tensor_mod_types_train, tensor_mod_types_valid = (
#             tokenized_train["mod_types"].long(),
#             tokenized_valid["mod_types"].long(),
#         )

#     if sort_seq_batch:
#         train_sort_ids = np.argsort(train_batch_labels)
#         input_gene_ids_train = input_gene_ids_train[train_sort_ids]
#         input_values_train = input_values_train[train_sort_ids]
#         target_values_train = target_values_train[train_sort_ids]
#         tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
#         if config.task == "annotation":
#             tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]
#         if config.task == "multiomic":
#             tensor_mod_types_train = tensor_mod_types_train[train_sort_ids]
#         valid_sort_ids = np.argsort(valid_batch_labels)
#         input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
#         input_values_valid = input_values_valid[valid_sort_ids]
#         target_values_valid = target_values_valid[valid_sort_ids]
#         tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
#         if config.task == "annotation":
#             tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]
#         if config.task == "multiomic":
#             tensor_mod_types_valid = tensor_mod_types_valid[valid_sort_ids]

#     train_data_pt = {
#         "gene_ids": input_gene_ids_train,
#         "values": input_values_train,
#         "target_values": target_values_train,
#         "batch_labels": tensor_batch_labels_train,
#     }
#     valid_data_pt = {
#         "gene_ids": input_gene_ids_valid,
#         "values": input_values_valid,
#         "target_values": target_values_valid,
#         "batch_labels": tensor_batch_labels_valid,
#     }
#     if config.task == "annotation":
#         train_data_pt["celltype_labels"] = tensor_celltype_labels_train
#         valid_data_pt["celltype_labels"] = tensor_celltype_labels_valid
#     if config.task == "multiomic":
#         train_data_pt["mod_types"] = tensor_mod_types_train
#         valid_data_pt["mod_types"] = tensor_mod_types_valid
#     return train_data_pt, valid_data_pt


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def train(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    scaler,
    optimizer,
    scheduler,
    device,
    config,
    logger,
    epoch,
    use_wandb: Optional[bool]=True,
    discriminator: Optional[nn.Module] = None,
    criterion_adv: Optional[nn.Module] = None,
    optimizer_E: Optional[torch.optim.Optimizer] = None,
    optimizer_D: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Train the model for one epoch.
    """
    if use_wandb:
        import wandb

    model.train()
    (
        total_loss,
        total_gep,
        total_cls,
        total_cce,
        total_gepc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_gepc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        if config.task == "annotation":
            celltype_labels = batch_data["celltype_labels"].to(device)
        if config.task == "multiomic":
            mod_types = batch_data["mod_types"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.use_batch_labels or config.DSBN else None,
                CLS=config.CLS,
                MVC=config.GEPC,
                CCE=config.CCE,
                ECS=config.ESC,
                mod_types=mod_types if config.use_mod else None,
                do_sample=config.do_sample_in_train if config.explicit_zero_prob else False,
                # generative_training=False
            )

            masked_positions = input_values.eq(
                config.mask_value
            )  # the postions to predict
            loss = 0.0
            metrics_to_log = {}
            if config.GEP:
                loss_gep = criterion_gep_gepc(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_gep
                metrics_to_log = {"train/gep": loss_gep.item()}
            
            if config.GEP and config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})

            if config.GEPC:
                loss_gepc = criterion_gep_gepc(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})

            if config.GEPC and config.explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )

            if config.CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            
            if config.CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})

            if (config.ESC) and (config.ecs_thres > 0):
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})

            if config.DAR: # domain adversarial loss   
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if config.ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.use_batch_labels or config.DSBN else None,
                CLS=config.CLS,
                MVC=config.GEPC,
                CCE=config.CCE,
                ECS=config.ESC,
                do_sample=config.do_sample_in_train,
                #generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > config.adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > config.adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()


        if use_wandb:
            wandb.log(metrics_to_log)

        if config.GEP:
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                metrics_to_log.update({"train/mre": mre.item()})


        total_loss += loss.item()
        total_gep += loss_gep.item() if config.GEP else 0.0
        total_cls += loss_cls.item() if config.CLS else 0.0
        total_cce += loss_cce.item() if config.CCE else 0.0
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_error += mre.item() if config.GEP else 0.0
        total_ecs += loss_ecs.item() if config.ESC else 0.0
        total_dab += loss_dab.item() if config.DAR else 0.0
        total_adv_E += loss_adv_E.item() if config.ADV else 0.0
        total_adv_D += loss_adv_D.item() if config.ADV else 0.0
        total_zero_log_prob += (
            loss_zero_log_prob.item()
            if config.GEP and config.explicit_zero_prob
            else 0.0
        )
        total_gepc_zero_log_prob += (
            loss_gepc_zero_log_prob.item()
            if config.GEPC and config.explicit_zero_prob
            else 0.0
        )
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_gep = total_gep / log_interval if config.GEP else 0.0
            cur_cls = total_cls / log_interval if config.CLS else 0.0
            cur_cce = total_cce / log_interval if config.CCE else 0.0
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval if config.GEP else 0.0
            cur_ecs = total_ecs / log_interval if config.ESC else 0.0
            cur_dab = total_dab / log_interval if config.DAR else 0.0
            cur_adv_E = total_adv_E / log_interval if config.ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if config.ADV else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if config.explicit_zero_prob else 0.0
            )
            cur_gepc_zero_log_prob = (
                total_gepc_zero_log_prob / log_interval
                if config.GEPC and config.explicit_zero_prob
                else 0.0
            )
            # cur_error = total_error / log_interval

            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"gep {cur_gep:5.2f} |" if config.GEP else "")
                + (f"cls {cur_cls:5.2f} | " if config.CLS else "")
                + (f"cce {cur_cce:5.2f} | " if config.CCE else "")
                + (f"err {cur_error:5.2f} | " if config.GEP else "")
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                + (f"ecs {cur_ecs:5.2f} |" if config.ESC else "")
                + (f"dar {cur_dab:5.2f} |" if config.DAR else "")
                + (f"adv_E {cur_adv_E:5.2f} |" if config.ADV else "")
                + (f"adv_D {cur_adv_D:5.2f} |" if config.ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if config.explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_gepc_zero_log_prob:5.2f} |"
                    if config.GEPC and config.explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_gep = 0
            total_cls = 0
            total_cce = 0
            total_gepc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_gepc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis(use_wandb=True):
    if use_wandb:
        import wandb
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/cls", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mvc", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mlm", summary="min", step_metric="epoch")
        wandb.define_metric("valid/ecs", summary="min", step_metric="epoch")
        wandb.define_metric("valid/cls_accuracy", summary="max", step_metric="epoch")
        wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/loss", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    device,
    config,
    epoch,
    use_wandb: bool =True,
    ret_dict: bool = False,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    if use_wandb:
        import wandb

    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_mlm_loss = 0.0
    total_mlm_error = 0.0
    total_gepc_loss = 0.0
    total_cls_accuracy = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            if config.task == "annotation":
                celltype_labels = batch_data["celltype_labels"].to(device)
            if config.task == "multiomic":
                mod_types = batch_data["mod_types"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS,
                    MVC=config.GEPC,
                    ECS=config.ESC,
                    mod_types=mod_types if config.use_mod else None,
                    # do_sample=do_sample_in_train,
                    # generative_training = False,
                )
                
                loss = torch.tensor(0,dtype=torch.float32)
                if 'cls_output' in output_dict:
                    output_values = output_dict["cls_output"]
                    cls_loss = criterion_cls(output_values, celltype_labels)
                    cls_accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                    
                else:
                    cls_loss = torch.tensor(0, dtype=torch.float32)
                    cls_accuracy = torch.tensor(0, dtype=torch.float32)

                if 'mlm_output' in output_dict:
                    masked_positions = input_values.eq(config.mask_value)
                    output_values = output_dict["mlm_output"]
                    mlm_loss = criterion_gep_gepc(
                        output_values, target_values, masked_positions
                    )

                    mlm_error = masked_relative_error(
                        output_values, target_values, masked_positions
                    )
                else:
                    mlm_loss = torch.tensor(0,dtype=torch.float32)
                    mlm_error = torch.tensor(0,dtype=torch.float32)

                if 'mvc_output' in output_dict:
                    output_values = output_dict["mvc_output"]
                    loss_gepc = criterion_gep_gepc(output_values, target_values, masked_positions)
                else:
                    loss_gepc = torch.tensor(0,dtype=torch.float32)



                if ('dab_output' in output_dict) and config.DAR:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                if config.task == "annotation":
                    # output_values = output_dict["cls_output"]
                    # loss = criterion_cls(output_values, celltype_labels)
                    loss = loss + cls_loss
                    # loss += cls_loss
                    tmp_error = torch.tensor(1 - cls_accuracy / len(input_gene_ids),dtype=torch.float32)
                    tmp_cls_accuracy = cls_accuracy

                elif config.task in ["integration", "multiomic"]:
                    # output_values = output_dict["mlm_output"]
                    # masked_positions = input_values.eq(config.mask_value)
                    # loss = criterion_gep_gepc(
                    #     output_values, target_values, masked_positions
                    # )
                    # loss += mlm_loss
                    loss = loss + mlm_loss
                    tmp_error = mlm_error
                    tmp_cls_accuracy = 0.0
                else:
                    loss = loss + mlm_loss
                    tmp_error = mlm_error
                    tmp_cls_accuracy = 0.0
                    

                # if config.DAR:
                #     loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_error += tmp_error.item() * len(input_gene_ids)
            total_loss += loss.item() * len(input_gene_ids)
            total_cls_loss += cls_loss.item() * len(input_gene_ids)
            total_mlm_loss += mlm_loss.item() * len(input_gene_ids)
            total_cls_accuracy += tmp_cls_accuracy * len(input_gene_ids)
            total_gepc_loss += loss_gepc.item() * len(input_gene_ids)
            
            # total_error += masked_relative_error(
            #     output_values, target_values, masked_positions
            # ).item() * len(input_gene_ids)
            
            if config.DAR:
                total_dab += (
                    loss_dab.item() * len(input_gene_ids) if config.DAR else 0.0
                )
            else:
                total_dab = 0

            total_num += len(input_gene_ids)

    if use_wandb:
        wandb.log(
            {
                "valid/loss": (total_loss + config.dab_weight * total_dab) / total_num,
                "valid/mse": total_loss / total_num,
                "valid/mre": total_error / total_num,
                "valid/cls": total_cls_loss / total_num,
                "valid/mlm": total_mlm_loss / total_num,
                "valid/cls_accuracy": total_cls_accuracy / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
                    / total_num,
                "epoch": epoch,
            },
        )
    if ret_dict:
        return {
            "loss": total_loss / total_num,
            "mse": total_loss / total_num,
            "mre": total_error / total_num,
            "cls": total_cls_loss / total_num,
            "gep": total_mlm_loss / total_num,
            "mlm": total_mlm_loss / total_num,
            'gepc': total_gepc_loss / total_num,
            'mvc': total_gepc_loss / total_num,
            "cls_accuracy": total_cls_accuracy / total_num,
            "dab": total_dab / total_num,
            "sum_mse_dab": (total_loss + config.dab_weight * total_dab) / total_num,
        }

    return total_loss / total_num


def predict(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    config,
    device,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_labels or config.DSBN
                    else None,
                    CLS=config.CLS,
                    MVC=config.GEPC,
                    ECS=config.ESC,
                )

                output_values = output_dict["cls_output"]
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)

    return np.concatenate(predictions, axis=0)


# %% inference
def test(
    model: nn.Module, adata: DataLoader, gene_ids, vocab, config, device, logger
) -> float:
    all_counts = (
        adata.layers[config.input_layer_key].A
        if issparse(adata.layers[config.input_layer_key])
        else adata.layers[config.input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=config.include_zero_gene,
    )


    # Taken from eval_testdata
    all_gene_ids, all_values = tokenized_test["genes"], tokenized_test["values"]
    src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
        cell_embeddings = model.encode_batch(
            all_gene_ids,
            all_values.float(),
            src_key_padding_mask=src_key_padding_mask,
            batch_size=config.batch_size,
            batch_labels=torch.from_numpy(batch_ids).long()
            if config.DSBN or config.DAR or config.use_batch_labels
            else None,
            time_step=0,
            return_np=True,
        )
    cell_embeddings = cell_embeddings / np.linalg.norm(
        cell_embeddings, axis=1, keepdims=True
    )

    # adata.obsm["X_scGPT"] = cell_embeddings
    #### end of eval_testdata

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), config.batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    predictions = predict(
        model,
        test_loader,
        vocab,
        config,
        device,
    )

    # compute accuracy, precision, recall, f1
    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")
    micro_f1 = f1_score(celltypes_labels, predictions, average="micro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
        "test/micro_f1": micro_f1,
    }

    return predictions, celltypes_labels, results, cell_embeddings


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    gene_ids,
    vocab,
    config,
    logger,
    criterion_gep_gepc=None,
    criterion_dab=None,
    criterion_cls=None,
    device= None,
    include_types: List[str] = ["cls"],
    subset_label: str = "Test",
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()
    if config.datasubset_label not in adata_t.obs:
        adata_t = adata_t.copy()
    else:
        if subset_label not in adata_t.obs[config.datasubset_label].unique():
            logger.error(f"subset_label {subset_label} not found in adata_t")
            return None
        adata_t = adata_t[adata_t.obs[config.datasubset_label] == subset_label].copy()

    all_counts = (
        adata_t.layers[config.input_layer_key].A
        if issparse(adata_t.layers[config.input_layer_key])
        else adata_t.layers[config.input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    celltypes_id_labels = adata_t.obs["celltype_id"].tolist()
    celltypes_id_labels = np.array(celltypes_id_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=config.max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=config.include_zero_gene,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])

        # Added by Jonah so that the model can be evaluated on the test data
        data_pt = prepare_dataset(tokenized_data=tokenized_all,
                                    batch_labels=batch_ids,
                                    config=config,
                                    celltype_labels=celltypes_id_labels,
                                    sort_seq_batch=config.per_seq_batch_sample,
                                    epoch=0)

        data_loader = prepare_dataloader(
            data_pt=data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            intra_domain_shuffle=False,
            per_seq_batch_sample=config.per_seq_batch_sample)


        eval_metrics = evaluate(
            model,
            data_loader,
            vocab,
            criterion_gep_gepc=criterion_gep_gepc if config.GEP and config.GEPC else None,
            criterion_dab=criterion_dab if config.DAR else None,
            criterion_cls=criterion_cls if config.CLS else None,
            device=device,
            config=config,
            epoch=0,
            use_wandb=False,
            ret_dict=True,
            )

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long()
                if config.DSBN or config.DAR or config.use_batch_labels
                else None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            print('eval scib metrics')
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        results.update(eval_metrics)

        print('sc.pp.neighbors')
        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results
