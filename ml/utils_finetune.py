# %%
import pandas as pd
import numpy as np
import neptune
import torch
from neptune.utils import stringify_unsupported
from misc import save_json, load_json, get_clean_batch_sz
from utils_neptune import get_latest_dataset, check_neptune_existance, check_if_path_in_struc, convert_neptune_kwargs
from models import get_encoder, get_head, MultiHead, create_model_wrapper, create_pytorch_model_from_info, CompoundModel
from train4 import train_compound_model, create_dataloaders_old, CompoundDataset
import os
import shutil
from collections import defaultdict
import re
import argparse

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'

# %%


def run_model_wrapper(data_dir, params, output_dir=None, 
                      train_name='train', prefix='training_run',
                      eval_name_list=['val'], eval_params_list=None, 
                      run_dict=None):
    """
    Runs the model training and evaluation pipeline.

    Args:
        data_dir (str): The directory path where the input data is stored.
        params (dict): A dictionary containing the model parameters.
        output_dir (str, optional): The directory path where the output models will be saved. 
            If not provided, a default directory will be used.
        train_name (str, optional): The name of the training dataset. Default is 'train'.
        prefix (str, optional): The prefix to be used for tracking and saving the models. 
            Default is 'training_run'.
        eval_name_list (list, optional): A list of names of the evaluation datasets. 
            Default is ['val'].
        eval_params_list (list, optional): A list of dictionaries containing evaluation parameters. 
            Each dictionary can contain 'y_col_name', 'y_cols', and 'y_head' keys. 
            Default is None.
        run_dict (neptune.metadata_containers.run.Run or neptune.handler.Handler, optional): 
            An object representing the Neptune run. If provided, the models will be tracked and saved to Neptune. 
            Default is None.

    Returns:
        dict: A dictionary containing the evaluation metrics for each evaluation dataset and parameter combination.

    Raises:
        FileNotFoundError: If the encoder_info.json file is not found in the saved_model_dir.

    """

    if run_dict is None:
        run_dict = {}

    if isinstance(run_dict, neptune.metadata_containers.run.Run) or isinstance(run_dict, neptune.handler.Handler):
        print('Using Neptune')
        use_neptune= True
        #TODO: check if the models are already trained on neptune
        run_struc = run_dict.get_structure()
        download_models_from_neptune = check_if_path_in_struc(run_struc,f'{prefix}/models/encoder_info')
        # download_models_from_neptune = check_neptune_existance(run_dict,f'{prefix}/models/encoder_info')


    if use_neptune:
    #     run_dict[f'{prefix}/dataset'].track_files(data_dir)
    #     run_dict[f'{prefix}/model_name'] = 'Model2925'

        # default_params = run_dict['params'].fetch()
        default_params = run_dict['params/train_kwargs'].fetch()
        default_params = convert_neptune_kwargs(default_params)
        # find the difference between the default params and the current params
        params_diff = {}
        # for k,v in params.items():
        for k,v in params['train_kwargs'].items():
            if isinstance(v,dict):
                for kk,vv in v.items():
                    if default_params.get(k) is None:
                        params_diff[k] = v
                    else:
                        if default_params[k].get(kk) != vv:
                            params_diff[k] = v
            else:
                if default_params.get(k) != v:
                    params_diff[k] = v

        # run_dict[f'{prefix}/params_diff'] = stringify_unsupported(params_diff)
        run_dict[f'{prefix}/params_diff/train_kwargs'] = stringify_unsupported(params_diff)
        
        # run_dict[f'dataset'].track_files(data_dir)
        # run_dict[f'model_name'] = 'Model2925'
        # run_dict[f'params'] = stringify_unsupported(params)

    if eval_params_list is None:
        eval_params_list = [{}]

    if output_dir is None:
        output_dir = os.path.expanduser('~/TEMP_MODELS')

    X_filename = 'X_finetune'
    y_filename = 'y_finetune'
    saved_model_dir = os.path.join(output_dir,prefix,'models')        
    os.makedirs(saved_model_dir,exist_ok=True)
    task_components_dict = params['task_kwargs']
    train_kwargs = params['train_kwargs']

    if (not os.path.exists(f'{saved_model_dir}/encoder_info.json')) and (use_neptune) and (download_models_from_neptune): 
        run_dict[f'{prefix}/models/encoder_state'].download(f'{saved_model_dir}/encoder_state.pt')
        run_dict[f'{prefix}/models/encoder_info'].download(f'{saved_model_dir}/encoder_info.json')
        run_dict[f'{prefix}/models/head_state'].download(f'{saved_model_dir}/head_state.pt')
        run_dict[f'{prefix}/models/head_info'].download(f'{saved_model_dir}/head_info.json')
        if os.path.exists(f'{saved_model_dir}/adv_info.json'):
            run_dict[f'{prefix}/models/adv_state'].download(f'{saved_model_dir}/adv_state.pt')
            run_dict[f'{prefix}/models/adv_info'].download(f'{saved_model_dir}/adv_info.json')
    
    if os.path.exists(f'{saved_model_dir}/encoder_info.json'):
        encoder = create_model_wrapper(f'{saved_model_dir}/encoder_info.json',f'{saved_model_dir}/encoder_state.pt')
        head = create_model_wrapper(f'{saved_model_dir}/head_info.json',f'{saved_model_dir}/head_state.pt',is_encoder=False)
        if os.path.exists(f'{saved_model_dir}/adv_info.json'):
            adv = create_model_wrapper(f'{saved_model_dir}/adv_info.json',f'{saved_model_dir}/adv_state.pt',is_encoder=False)
        else:
            adv = MultiHead([])

    else:
        X_data_train = pd.read_csv(f'{data_dir}/{X_filename}_{train_name}.csv', index_col=0)
        y_data_train = pd.read_csv(f'{data_dir}/{y_filename}_{train_name}.csv', index_col=0)

        _, encoder, head, adv = fit_model_wrapper(X=X_data_train,
                                                y=y_data_train,
                                                task_components_dict=task_components_dict,
                                                run_dict=run_dict[prefix],
                                                **train_kwargs)

        save_model_wrapper(encoder, head, adv, 
                           save_dir=saved_model_dir,
                           run_dict=run_dict,
                           prefix=prefix)


    metrics = defaultdict(dict)
    for eval_name in eval_name_list:
        X_data_eval = pd.read_csv(f'{data_dir}/{X_filename}_{eval_name}.csv', index_col=0)
        y_data_eval = pd.read_csv(f'{data_dir}/{y_filename}_{eval_name}.csv', index_col=0)

        for eval_params in eval_params_list:
            y_col_name = eval_params.get('y_col_name',None)
            y_cols = eval_params.get('y_cols',None)
            y_head = eval_params.get('y_head',None)
            if y_cols is None:
                y_cols = params['task_kwargs']['y_head_cols']

            try:

                if y_col_name is None:
                    metrics[f'{eval_name}' ].update(evaluate_model_wrapper(encoder, head, adv, X_data_eval, y_data_eval,
                                                                        y_cols=y_cols,
                                                                        y_head=y_head))
                else:
                    metrics[f'{eval_name}__{y_col_name}'].update(evaluate_model_wrapper(encoder, head, adv, X_data_eval, y_data_eval,
                                                                                        y_cols=y_cols,
                                                                                        y_head=y_head))
            except ValueError as e:
                print(f'Error: {e}')
                print(f'Error in {eval_name}__{y_col_name}')
                # metrics[f'{eval_name}__{y_col_name}'] = None
                
    if use_neptune:
        run_dict[f'{prefix}/metrics'] = metrics
        run_dict.wait()

    return metrics

############################################################


### Function to get the encoder
def get_pretrained_encoder(dropout_rate=None, use_rand_init=False, load_dir=None, verbose=False):
    """
    Retrieves a pretrained encoder model.

    Args:
        dropout_rate (float, optional): The dropout rate to be set in the encoder model. Defaults to None.
        use_rand_init (bool, optional): Whether to use random initialization for the encoder model. Defaults to False.
        load_dir (str, optional): The directory path to load the pretrained model from. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        encoder (object): The pretrained encoder model.

    Raises:
        ValueError: If the 'hidden_size_mult' key is missing in the encoder_kwargs dictionary.

    """
    
    
    if load_dir is None:
        load_dir = os.path.expanduser('~/PRETRAINED_MODELS/2925')
        os.makedirs(load_dir,exist_ok=True)

    encoder_kwargs_path = os.path.join(load_dir,'encoder_kwargs.json')
    encoder_state_path =  os.path.join(load_dir,'encoder_state.pt')

    if (not os.path.exists(encoder_kwargs_path)) or (not os.path.exists(encoder_state_path)):

        neptune_model = neptune.init_model(project='revivemed/Survival-RCC',
            api_token= NEPTUNE_API_TOKEN,
            with_id='SUR-MOD',
            mode="read-only")

        if not os.path.exists(encoder_state_path):
            neptune_model['model/encoder_state'].download(encoder_state_path)
        if not os.path.exists(encoder_kwargs_path):
            encoder_kwargs = neptune_model['original_kwargs/encoder_kwargs'].fetch()
            if 'input_size' not in encoder_kwargs:
                encoder_kwargs['input_size'] = 2736
            if 'kind' not in encoder_kwargs:
                encoder_kwargs['kind'] = neptune_model['original_kwargs/encoder_kind'].fetch()
            if 'hidden_size' not in encoder_kwargs:
                if 'hidden_size_mult' in encoder_kwargs:
                    latent_size = encoder_kwargs['latent_size']
                    encoder_kwargs['hidden_size'] = int(encoder_kwargs['hidden_size_mult']*latent_size)
                else:
                    raise ValueError()
                # remove the hidden_size_mult key
                encoder_kwargs.pop('hidden_size_mult')
            save_json(encoder_kwargs,encoder_kwargs_path)

        neptune_model.stop()

    encoder_kwargs = load_json(encoder_kwargs_path)
    if (dropout_rate is not None):
        if verbose: print('Setting dropout rate to',dropout_rate)
        encoder_kwargs['dropout_rate'] = dropout_rate

    encoder = get_encoder(**encoder_kwargs)

    if not use_rand_init:
        encoder.load_state_dict(torch.load(encoder_state_path))

    return encoder



#### Function to get the model heads
def get_model_heads(head_kwargs_dict, backup_input_size=None, verbose=False):
    """
    Returns a MultiHead object based on the provided head_kwargs_dict.

    Args:
        head_kwargs_dict (dict or list): A dictionary or list of dictionaries containing the keyword arguments for each head.
        backup_input_size (int, optional): The backup input size to use if 'input_size' is not specified in the head_kwargs. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        MultiHead: A MultiHead object containing the heads created based on the provided head_kwargs_dict.

    Raises:
        ValueError: If the head_kwargs_dict is not a dictionary or a list.

    Example:
        head_kwargs_dict = {
            'head1': {'input_size': 100, 'output_size': 10},
            'head2': {'input_size': 200, 'output_size': 20}
        }
        heads = get_model_heads(head_kwargs_dict, backup_input_size=50, verbose=True)
    """
    
    # Function implementation goes here
    
    if head_kwargs_dict is None:
        return MultiHead([]) #get_head(kind='Dummy')
    
    if isinstance(head_kwargs_dict,dict):
        head_kwargs_list = [head_kwargs_dict[k] for k in head_kwargs_dict.keys()]
    elif isinstance(head_kwargs_dict,list):
        head_kwargs_list = head_kwargs_dict
    else:
        raise ValueError(f'Invalid head_kwargs_dict type: {type(head_kwargs_dict)}')

    head_list = []
    if len(head_kwargs_list) == 0:
       return MultiHead([])

    for h_kwargs in head_kwargs_list:
            
        if 'input_size' not in h_kwargs:
            if backup_input_size is None:
                raise ValueError('backup_input_size is None')
            if verbose: print('Setting input_size to',backup_input_size)
            h_kwargs['input_size'] = backup_input_size

        head = get_head(**h_kwargs)
        head_list.append(head)


    head = MultiHead(head_list)
    return head


### Function to build the model components for fine-tuning
def build_model_components(head_kwargs_dict, adv_kwargs_dict=None, dropout_rate=None, use_rand_init=False):
    """
    Builds the model components for fine-tuning.

    Args:
        head_kwargs_dict (dict): A dictionary containing the keyword arguments for the head model.
        adv_kwargs_dict (dict, optional): A dictionary containing the keyword arguments for the adversary model. Defaults to None.
        dropout_rate (float, optional): The dropout rate. Defaults to None.
        use_rand_init (bool, optional): Whether to use random initialization. Defaults to False.

    Returns:
        tuple: A tuple containing the encoder model, head model, and adversary model.
    """
    encoder = get_pretrained_encoder(dropout_rate=dropout_rate, use_rand_init=use_rand_init)

    # TODO: We need better handling of head input size to account for the size of the other_vars
    head = get_model_heads(head_kwargs_dict, backup_input_size=encoder.latent_size + 1)
    if head.kind == 'MultiHead':
        head.name = 'HEAD'

    adv = get_model_heads(adv_kwargs_dict, backup_input_size=encoder.latent_size)
    if adv.kind == 'MultiHead':
        adv.name = 'ADVERSARY'

    return encoder, head, adv






def fit_model_wrapper(X, y, task_components_dict={}, run_dict={}, **train_kwargs):
    """
    Fits a compound model using the provided data and model components.

    Parameters:
    - X (numpy.ndarray): The input data for training the model.
    - y (pandas.DataFrame): The target variable(s) for training the model.
    - task_components_dict (dict): A dictionary containing the model component defaults.
    - run_dict (neptune.metadata_containers.run.Run or neptune.handler.Handler): 
        An optional Neptune run object for recording the model fitting.
    - **train_kwargs: Additional keyword arguments for training the model.

    Returns:
    - run_dict (neptune.metadata_containers.run.Run or neptune.handler.Handler): 
        The updated Neptune run object.
    - encoder: The trained encoder component of the compound model.
    - head: The trained head component of the compound model.
    - adv: The trained adversary component of the compound model.
    """
    
    assert isinstance(task_components_dict,dict), 'task_components_dict should be a dictionary'

    if isinstance(run_dict, neptune.metadata_containers.run.Run) or isinstance(run_dict, neptune.handler.Handler):
        print('Record the model fitting to Neptune')
        use_neptune= True
    else:
        use_neptune = False
        
    ### Model Component Defaults
    y_head_cols = task_components_dict.get('y_head_cols',None)
    y_adv_cols = task_components_dict.get('y_adv_cols',None)
    head_kwargs_dict = task_components_dict.get('head_kwargs_dict',None)
    adv_kwargs_dict = task_components_dict.get('adv_kwargs_dict',None)

    if y_head_cols is None:
        # select all of the numeric columns
        y_head_cols = list(y.select_dtypes(include=[np.number]).columns)

    if y_adv_cols is None:
        y_adv_cols = []

    ### Train Defaults
    dropout_rate = train_kwargs.get('dropout_rate', None)
    use_rand_init = train_kwargs.get('use_rand_init', False)
    batch_size = train_kwargs.get('batch_size', 64)
    holdout_frac = train_kwargs.get('holdout_frac', 0)
    early_stopping_patience = train_kwargs.get('early_stopping_patience', 0)
    scheduler_kind = train_kwargs.get('scheduler_kind', None)
    train_name = train_kwargs.get('train_name', 'train')

    ### Prepare the Data Loader
    X_size = X.shape[1]
    if (holdout_frac > 0) and (early_stopping_patience < 1) and (scheduler_kind is None):
        # raise ValueError('holdout_frac > 0 and early_stopping_patience < 1 is not recommended')
        print('holdout_frac > 0 and early_stopping_patience < 1 is not recommended, set hold out frac to 0')
        print('UNLESS you are using a scheduler, in which case the holdout_frac is used for the scheduler')
        holdout_frac = 0
        batch_size = get_clean_batch_sz(X_size, batch_size)
    else:
        batch_size = get_clean_batch_sz(X_size*(1-holdout_frac), batch_size)

    train_dataset = CompoundDataset(X,y[y_head_cols], y[y_adv_cols])
    # stratify on the adversarial column (stratify=2)
    # this is probably not the most memory effecient method, would be better to do stratification before creating the dataset
    # train_loader_dct = create_dataloaders(train_dataset, batch_size, holdout_frac, set_name=train_name, stratify=2)
    train_loader_dct = create_dataloaders_old(train_dataset, batch_size, holdout_frac, set_name=train_name)


    ### Build the Model Components
    encoder, head, adv = build_model_components(head_kwargs_dict=head_kwargs_dict,
                                                adv_kwargs_dict=adv_kwargs_dict,
                                                dropout_rate=dropout_rate,
                                                use_rand_init=use_rand_init)

    if train_dataset is not None:
        #TODO load the class weights from the model info json
        head.update_class_weights(train_dataset.y_head)
        adv.update_class_weights(train_dataset.y_adv)

    if len(adv.heads)==0:
        train_kwargs['adversary_weight'] = 0

    ### Train the Model
    encoder, head, adv = train_compound_model(train_loader_dct, 
                                            encoder, head, adv, 
                                            run=run_dict, **train_kwargs)
    if encoder is None:
        raise ValueError('Encoder is None after training, training failed')

    
    return run_dict, encoder, head, adv



def save_model_wrapper(encoder, head, adv, save_dir=None, run_dict={}, prefix='training_run'):
    """
    Saves the encoder, head, and adversary models to the specified directory.

    Args:
        encoder (Encoder): The encoder model.
        head (Head): The head model.
        adv (Adversary): The adversary model.
        save_dir (str, optional): The directory to save the models. If not provided, a temporary directory will be used. Defaults to None.
        run_dict (dict, optional): A dictionary containing information about the Neptune run. Defaults to {}.
        prefix (str, optional): The prefix to use for the Neptune run. Defaults to 'training_run'.

    Returns:
        dict: A dictionary containing information about the Neptune run.

    Raises:
        NotImplementedError: If `upload_models_to_gcp` is set to True.

    Note:
        - If `save_dir` is not provided, a temporary directory will be created at '~/TEMP_MODELS'.
        - If `run_dict` is an instance of `neptune.metadata_containers.run.Run` or `neptune.handler.Handler`, the models will be saved to Neptune.
        - If `upload_models_to_gcp` is set to True, an exception will be raised as this functionality is not implemented.
    """
    

    if isinstance(run_dict, neptune.metadata_containers.run.Run) or isinstance(run_dict, neptune.handler.Handler):
        print('Save models to Neptune')
        use_neptune= True
    else:
        use_neptune = False

    delete_after_upload = False
    if save_dir is None:
        save_dir = os.path.expanduser('~/TEMP_MODELS')
        if use_neptune:
            delete_after_upload = True
            if os.path.exists(save_dir):
                # delete the directory
                shutil.rmtree(save_dir)
        os.makedirs(save_dir,exist_ok=True)


    upload_models_to_gcp = False    

    encoder.save_state_to_path(save_dir,save_name='encoder_state.pt')
    encoder.save_info(save_dir,save_name='encoder_info.json')
    head.save_state_to_path(save_dir,save_name='head_state.pt')
    head.save_info(save_dir,save_name='head_info.json')
    adv.save_state_to_path(save_dir,save_name='adversary_state.pt')
    adv.save_info(save_dir,save_name='adversary_info.json')

    # torch.save(head.state_dict(), f'{save_dir}/{setup_id}_head_state_dict.pth')
    # torch.save(adv.state_dict(), f'{save_dir}/{setup_id}_adv_state_dict.pth')
    if use_neptune:
        run_dict[f'{prefix}/models/encoder_state'].upload(f'{save_dir}/encoder_state.pt')
        run_dict[f'{prefix}/models/encoder_info'].upload(f'{save_dir}/encoder_info.json')
        run_dict[f'{prefix}/models/head_state'].upload(f'{save_dir}/head_state.pt')
        run_dict[f'{prefix}/models/head_info'].upload(f'{save_dir}/head_info.json')
        run_dict[f'{prefix}/models/adv_state'].upload(f'{save_dir}/adv_state.pt')
        run_dict[f'{prefix}/models/adv_info'].upload(f'{save_dir}/adv_info.json')
        run_dict.wait()

    if upload_models_to_gcp:
        raise NotImplementedError('upload_models_to_gcp not implemented')

    if use_neptune and delete_after_upload:
        run_dict.wait()
        shutil.rmtree(save_dir)

    return run_dict



def evaluate_model_wrapper(encoder, head, adv, X_data_eval, y_data_eval, y_cols, y_head=None):
    """
    Wrapper function to evaluate a model.

    Parameters:
    - encoder: The encoder model.
    - head: The head model.
    - adv: The adversarial model.
    - X_data_eval: The input data for evaluation.
    - y_data_eval: The target data for evaluation.
    - y_cols: The columns to be used as target variables.
    - y_head: The name of the specific head to be used for evaluation.

    Returns:
    - The evaluation score of the model.

    Raises:
    - ValueError: If an invalid head name or y_cols length is provided.
    """

    if y_head is None:
        chosen_head = head
    else:
        multihead_name_list = head.heads_names
        if y_head not in multihead_name_list:
            raise ValueError(f'Invalid head name: {y_head}')
        chosen_head_idx = multihead_name_list.index(y_head)
        chosen_head = head.heads[chosen_head_idx]

        if isinstance(chosen_head.y_idx,list):
            if len(y_cols) != len(chosen_head.y_idx):
                raise ValueError(f'Invalid y_cols length: {len(y_cols)} vs {len(chosen_head.y_idx)}')
            if len(y_cols) == 1:
                chosen_head.y_idx = [0]
            else:
                chosen_head.y_idx = list(range(len(y_cols)))
        else:
            if len(y_cols) != 1:
                raise ValueError(f'Invalid y_cols length: {len(y_cols)} vs {len(chosen_head.y_idx)}')
            chosen_head.y_idx = 0

        # if len(y_cols) != len(chosen_head.y_idx):
        #     raise ValueError(f'Invalid y_cols length: {len(y_cols)} vs {len(chosen_head.y_idx)}')
        # if len(y_cols) == 1:
        #     chosen_head.y_idx = 0
        # else:
        #     chosen_head.y_idx = list(range(len(y_cols)))
        
    model = CompoundModel(encoder, chosen_head)
    skmodel = create_pytorch_model_from_info(full_model=model)

    return skmodel.score(X_data_eval.to_numpy(),y_data_eval[y_cols].to_numpy())


############################################################
############################################################
# %% CUUSTOM Functions Related to Determine Model Parameters 
############################################################

# default_sweep_kwargs = {
#     'holdout_frac': 0,
#     'head_hidden_layers': 0,
#     'encoder_kwargs__dropout_rate': 0.2,
#     'use_rand_init': False,
#     'train_kwargs__num_epochs': 30,
#     'train_kwargs__early_stopping_patience': 0,
#     'train_kwargs__learning_rate': 0.0005,
#     'train_kwargs__l2_reg_weight': 0.0005,
#     'train_kwargs__l1_reg_weight': 0.005,
#     'train_kwargs__noise_factor': 0.1,
#     'train_kwargs__weight_decay': 0,
#     'train_kwargs__adversary_weight': 1,
#     'train_kwargs__adversarial_start_epoch': 10,
#     'train_kwargs__encoder_weight': 0,
#     'train_kwargs__clip_grads_with_norm': False,
# }



def get_head_kwargs_by_desc(desc_str, num_hidden_layers=0, weight=1, y_cols=None):
    """
    Returns the head keyword arguments, y_head_cols, and plot_latent_space_cols based on the given description string.

    Parameters:
    - desc_str (str): The description string used to determine the head configuration.
    - num_hidden_layers (int): The number of hidden layers in the head. Default is 0.
    - weight (int): The weight of the head. Default is 1.
    - y_cols (list): The list of column names for the target variables. Default is None.

    Returns:
    - head_kwargs (dict): The keyword arguments for the head configuration.
    - y_head_cols (list): The column names for the target variables.
    - plot_latent_space_cols (list): The column names for plotting the latent space.

    Raises:
    - ValueError: If the given desc_str is unknown.

    """
    if (desc_str is None) or (desc_str == ''):
        return None, [], []
    
    if y_cols is None:
        y_cols = []

    if 'weight-' in desc_str:
        match = re.search(r'weight-(\d+)', desc_str)
        if match:
            weight = int(match.group(1))
            desc_str = desc_str.replace(match.group(0),'')

    if 'mskcc' in desc_str.lower():
        # if 'mskcc-ord' in desc_str.lower:
            # raise NotImplementedError('not implemented yet')
        y_head_cols = ['MSKCC BINARY']
        head_name = 'MSKCC'
        head_kind = 'Binary'
        num_classes = 2
        y_idx = 0
        plot_latent_space_cols = ['MSKCC']

    elif 'imdc' in desc_str.lower():
        y_head_cols = ['IMDC BINARY']
        head_name = 'IMDC'
        head_kind = 'Binary'
        num_classes = 2
        y_idx = 0
        plot_latent_space_cols = ['IMDC']

    elif 'nivo-benefit' in desc_str.lower():
        raise NotImplementedError()

    elif 'benefit' in desc_str.lower():
        y_head_cols = ['Benefit BINARY']
        head_name = 'Benefit'
        head_kind = 'Binary'
        num_classes = 2
        y_idx = 0
        plot_latent_space_cols = ['Benefit']

    elif 'both-os' in desc_str.lower():
        y_head_cols = ['OS','OS_Event']
        head_name = 'Both OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['OS']   

    elif 'both-pfs' in desc_str.lower():
        y_head_cols = ['PFS','PFS_Event']
        head_name = 'Both PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['PFS']        

    elif 'nivo-os' in desc_str.lower():
        y_head_cols = ['NIVO OS','OS_Event']
        head_name = 'NIVO OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['NIVO OS']   

    elif 'nivo-pfs' in desc_str.lower():
        y_head_cols = ['NIVO PFS','PFS_Event']
        head_name = 'NIVO PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['NIVO PFS']         

    elif 'ever-os' in desc_str.lower():
        y_head_cols = ['EVER OS','OS_Event']
        head_name = 'EVER OS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['EVER OS']   

    elif 'ever-pfs' in desc_str.lower():
        y_head_cols = ['EVER PFS','PFS_Event']
        head_name = 'EVER PFS'
        head_kind = 'Cox'
        num_classes = 1
        y_idx = [0,1]
        plot_latent_space_cols = ['EVER OS']            
    else:
        raise ValueError('Unknown desc_str:',desc_str)

    for col in y_head_cols:
        if col not in y_cols:
            y_cols.append(col)

    if len(y_head_cols) == 1:
        y_idx = y_cols.index(y_head_cols[0])
    else:
        y_idx = [y_cols.index(col) for col in y_head_cols]


    head_kwargs = {
            'kind': head_kind,
            'name': head_name,
            'weight': weight,
            'y_idx': y_idx,
            'hidden_size': 4,
            'num_hidden_layers': num_hidden_layers,
            'dropout_rate': 0,
            'activation': 'leakyrelu',
            'use_batch_norm': False,
            'num_classes': num_classes,
            }

    return head_kwargs, y_head_cols, plot_latent_space_cols


def parse_task_components_dict_from_str(desc_str,sweep_kwargs=None):

    if sweep_kwargs is None:
        sweep_kwargs = {}

    task_components_dict = {}

    clean_desc_str = desc_str
    if 'optuna_' in desc_str:
        clean_desc_str = clean_desc_str.replace('optuna_','')
    if 'Optimized_' in clean_desc_str:
        clean_desc_str = clean_desc_str.replace('Optimized_','')
    if '__' in clean_desc_str:
        clean_desc_str = clean_desc_str.split('__')[0]

    if 'ADV' in clean_desc_str:
        adv_desc_str = clean_desc_str.split('ADV')[1]
        head_desc_str = clean_desc_str.split('ADV')[0]
    else:
        adv_desc_str = ''
        head_desc_str = clean_desc_str

    y_head_cols = []
    head_kwargs_list = []
    plot_latent_space_cols = []
    
    if 'AND' in head_desc_str:
        head_desc_str_list = head_desc_str.split('AND')
    else:
        head_desc_str_list = [head_desc_str]

    head_hidden_layers = sweep_kwargs.get('head_hidden_layers',0)

    for h_desc in head_desc_str_list:
        head_weight = sweep_kwargs.get(f'{h_desc}__weight',1)
        head_kwargs, head_cols, plot_latent_space_head_cols = get_head_kwargs_by_desc(h_desc,
                                                                                    num_hidden_layers=head_hidden_layers,
                                                                                    weight=head_weight,y_cols=y_head_cols)
        if head_kwargs is None:
            continue
        head_kwargs_list.append(head_kwargs)
        for col in head_cols:
            if col not in y_head_cols:
                y_head_cols.append(col)

        for col in plot_latent_space_head_cols:
            if col not in plot_latent_space_cols:
                plot_latent_space_cols.append(col)

    y_adv_cols = []
    adv_kwargs_list = []

    if 'AND' in adv_desc_str:
        adv_desc_str_list = adv_desc_str.split('AND')
    else:
        adv_desc_str_list = [adv_desc_str]

    for a_desc in adv_desc_str_list:
        adv_weight = sweep_kwargs.get(f'{a_desc}__weight',1)
        adv_kwargs, adv_cols, plot_latent_space_adv_cols = get_head_kwargs_by_desc(a_desc,
                                                                                   num_hidden_layers=head_hidden_layers,
                                                                                    weight=adv_weight,y_cols=y_adv_cols)
        if adv_kwargs is None:
            continue
        adv_kwargs_list.append(adv_kwargs)
        for col in adv_cols:
            if col not in y_adv_cols:
                y_adv_cols.append(col)
        for col in plot_latent_space_adv_cols:
            if col not in plot_latent_space_cols:
                plot_latent_space_cols.append(col)

    task_components_dict = {
        'y_head_cols': y_head_cols,
        'y_adv_cols': y_adv_cols,
        'head_kwargs_dict': head_kwargs_list,
        'adv_kwargs_dict': adv_kwargs_list,
        'plot_latent_space_cols': plot_latent_space_cols
    }

    return task_components_dict



def get_params(desc_str,sweep_kwargs=None):

    task_components_dict = parse_task_components_dict_from_str(desc_str,sweep_kwargs)

    train_kwargs_list = [
        'use_rand_init',
        'batch_size',
        'holdout_frac',
        'early_stopping_patience',
        'scheduler_kind',
        'num_epochs',
        'learning_rate',
        'noise_factor',
        'weight_decay',
        'l2_reg_weight',
        'l1_reg_weight',
        'dropout_rate',
        'adversarial_start_epoch',
        'clip_grads_with_norm',
        'encoder_weight',
        'adversary_weight',
        'train_name'
    ]

    train_kwargs_dict = {k:v for k,v in sweep_kwargs.items() if k in train_kwargs_list}
    params = {
        'task_kwargs': task_components_dict,
        'train_kwargs': train_kwargs_dict
    }
    return params


def parse_sweep_kwargs_from_command_line():

    parser = argparse.ArgumentParser(description='Parse command line arguments for sweep kwargs')
    parser.add_argument('--use_randinit', action='store_true', help='Use random initialization')
    parser.add_argument('--holdout_frac', type=float, default=0, help='Holdout fraction')
    parser.add_argument('--head_hidden_layers', type=int, default=0, help='Number of hidden layers in the head')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0, help='L2 regularization weight')
    parser.add_argument('--l1_reg_weight', type=float, default=0.0, help='L1 regularization weight')
    parser.add_argument('--noise_factor', type=float, default=0.1, help='Noise factor')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--adversary_weight', type=float, default=1, help='Adversary weight')
    parser.add_argument('--adversarial_start_epoch', type=int, default=10, help='Adversarial start epoch')
    parser.add_argument('--encoder_weight', type=float, default=0, help='Encoder weight')
    # parser.add_argument('--clip_grads', action='store_true', help='Clip gradients with norm')
    parser.add_argument('--no_clip_grads', action='store_false', help='Clip gradients with norm')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_name', type=str, default='train', help='Training name')

    args = parser.parse_args()

    sweep_kwargs = {
        'use_rand_init': args.use_randinit,
        'holdout_frac': args.holdout_frac,
        'head_hidden_layers': args.head_hidden_layers,
        'dropout_rate': args.dropout_rate,
        'num_epochs': args.num_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'learning_rate': args.learning_rate,
        'l2_reg_weight': args.l2_reg_weight,
        'l1_reg_weight': args.l1_reg_weight,
        'noise_factor': args.noise_factor,
        'weight_decay': args.weight_decay,
        'adversary_weight': args.adversary_weight,
        'adversarial_start_epoch': args.adversarial_start_epoch,
        'encoder_weight': args.encoder_weight,
        # 'clip_grads_with_norm': args.clip_grads,
        'clip_grads_with_norm': args.no_clip_grads,
        'batch_size': args.batch_size,
        'train_name': args.train_name
    }

    return sweep_kwargs