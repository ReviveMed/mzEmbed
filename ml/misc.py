import os
import torch

def get_dropbox_dir():
    my_dir = '/Users/jonaheaton/Library/CloudStorage/Dropbox-ReviveMed/Jonah Eaton'
    if not os.path.exists(my_dir):
        my_dir = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton'
    if not os.path.exists(my_dir):
        raise ValueError('Dropbox directory not found')

    return my_dir


def normalize_loss(current_loss, loss_avg=1, beta=0, current_epoch=1):
    # normalize the loss by its average, useful when creating multi-output models
    if current_epoch < 0:
        new_loss = current_loss / loss_avg
        return new_loss, loss_avg
    if beta  == 0:
        loss_avg = (loss_avg*current_epoch + current_loss.item()) / (current_epoch + 1)
    elif beta < 0:
        loss_avg = 1
    else:
        loss_avg = beta * loss_avg + (1 - beta) * current_loss.item()

    new_loss = current_loss / loss_avg
    return new_loss, loss_avg


def round_to_even(n):
    if n % 2 == 0:
        return n
    else:
        return n + 1

def get_clean_batch_sz(len_dataset, org_batch_sz):
    # due to batch normalization, we want the batches to be as clean as possible
    curr_remainder = len_dataset % org_batch_sz
    max_iter = 100
    if org_batch_sz >= len_dataset:
        return org_batch_sz
    if (curr_remainder == 0) or (curr_remainder > org_batch_sz/2):
        return org_batch_sz
    else:
        batch_sz = org_batch_sz
        iter = 0
        while (curr_remainder != 0) and (curr_remainder < batch_sz/2) and (iter < max_iter):
            iter += 1
            if batch_sz < org_batch_sz/2:
                batch_sz = 2*org_batch_sz
            batch_sz -= 1
            curr_remainder = len_dataset % batch_sz
        if iter >= max_iter:
            print('Warning: Could not find a clean batch size')
        # print('old batch size:', org_batch_sz, 'new batch size:', batch_sz, 'remainder:', curr_remainder)
        return batch_sz