from main_finetune import finetune_run_wrapper


############## Layer methods ############

layer1_0 = {
    'noise_factor': 0.25,
    'learning_rate': 0.0007869775056037999,
    'l2_reg_weight': 1.0092405183765013e-05,
    'l1_reg_weight': 3.137204254745065e-05,
    'num_epochs': 87,
    'head_hidden_layers': 1,
    'EVER-OS__weight': 10.0,
    'NIVO-OS__weight': 5.0,
    'name': 'layer 1.0'
}

list_layerS = [layer1_0]

############## Layer-R methods ############

layerR1_0 = {
    'noise_factor': 0.1,
    'learning_rate': 0.0006221023998363983,
    'l2_reg_weight': 0.0,
    'l1_reg_weight': 0.0025635844524779894,
    'num_epochs': 93,
    'encoder_weight': 1.0,
    'dropout_rate': 0.3,
    'adversarial_start_epoch': 0,
    'adversary_weight': 1.0,
    'head_hidden_layers': 1,
    'EVER-OS__weight': 10.0,
    'NIVO-OS__weight': 4.0,
    'name': 'layer-R 1.0'
}

layerR1_1 = {
    'noise_factor': 0.1,
    'learning_rate': 0.0005332970366817577,
    'l2_reg_weight': 0.0,
    'l1_reg_weight': 0.002566981823368559,
    'weight_decay': 0.0024617292382149805,
    'num_epochs': 98,
    'encoder_weight': 1.0,
    'dropout_rate': 0.3,
    'adversarial_start_epoch': 0,
    'adversary_weight': 3.5,
    'head_weight': 10.0,
    'head_hidden_layers': 1,
    'name': 'layer-R 1.1'
}


layerR1_2 = {
    'noise_factor': 0.1,
    'learning_rate': 0.0005332970366817577,
    'l2_reg_weight': 0.0,
    'l1_reg_weight': 0.002566981823368559,
    'weight_decay': 0.0024617292382149805,
    'num_epochs': 98,
    'encoder_weight': 1.0,
    'dropout_rate': 0.3,
    'adversarial_start_epoch': 10,
    'adversary_weight': 3.5,
    'head_weight': 10.0,
    'head_hidden_layers': 1,
    'name': 'layer-R 1.2'
}

layerR2_0 = {
    'noise_factor': 0.2,
    'early_stopping_patience': 10,
    'learning_rate': 0.0007425953363382413,
    'l2_reg_weight': 0.00013446262226818383,
    'l1_reg_weight': 0.0007243134748616204,
    'num_epochs': 59,
    'encoder_weight': 1.0,
    'auxillary_weight': 4.5,
    'dropout_rate': 0.3,
    'adversarial_start_epoch': 0,
    'adversary_weight': 1.0,
    'head_hidden_layers': 1,
    'batch_size': 64,
    'clean_batch': True,
    'remove_nans': False,
    'name': 'layer-R 2.0'
}


list_layerR = [layerR1_0,layerR2_0,layerR1_1,layerR1_2]


######## Basic methods ############

basic1_0 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.000541422,
    'noise_factor': 0.2,
    'num_epochs': 71,
    'batch_size': 64,
    'head_hidden_layers': 0,
    'name' : 'basic 1.0'
}

basic1_1 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.000541422,
    'noise_factor': 0.2,
    'num_epochs': 71,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'name': 'basic 1.1'
}

basic1_2 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.000541422,
    'noise_factor': 0.2,
    'num_epochs': 71,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'remove_nans': True,
    'name': 'basic 1.2'
}

basic2_0 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.0013,
    'noise_factor': 0.2,
    'num_epochs': 98,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'weight_decay': 0.000016,
    'name': 'basic 2.0'
}

basic2_1 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.0013,
    'noise_factor': 0.2,
    'num_epochs': 98,
    'batch_size': 32,
    'clean_batch': False,
    'head_hidden_layers': 0,
    'weight_decay': 0.000016,
    'remove_nans': True,
    'name': 'basic 2.1'
}

list_basicS = [basic1_0, basic1_1, basic1_2, basic2_0, basic2_1]

###### Basic-R methods ############

basicR1_0 = {
    'dropout_rate': 0.4,
    'encoder_weight': 0.75,
    'learning_rate': 0.0021,
    'noise_factor': 0.25,
    'num_epochs': 38,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'weight_decay': 0.0,
    'remove_nans': True,
    'name': 'basic-R 1.0'
}

basicR1_1 = {
    'dropout_rate': 0.4,
    'encoder_weight': 0.75,
    'learning_rate': 0.0021,
    'noise_factor': 0.25,
    'num_epochs': 38,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'weight_decay': 0.0,
    'remove_nans': False,
    'name': 'basic-R 1.1'
}

list_basicR = [basicR1_0, basicR1_1]



all_methods = list_layerS + list_layerR + list_basicS + list_basicR





desc_str_list1 = ['Both-OS','NIVO-OS','EVER-OS','NIVO-OS AND EVER-OS','IMDC','MSKCC','NIVO-OS ADV EVER-OS']
desc_str_list2 = ['IMDC-ord','IMDC-multi','both-OS AUX Benefit','MSKCC-ord','MSKCC-multi','EVER-PFS','NIVO-PFS','both-PFS']

# user_kwargs = parse_sweep_kwargs_from_command_line()
# method6, method7, method8, method9,method2
for method in  all_methods:
    for desc_str in desc_str_list2:
        for use_randinit in [True,False]:
            user_kwargs = {k:v for k,v in method.items()}
            user_kwargs['use_rand_init'] = use_randinit
            user_kwargs['desc_str'] = desc_str
            try:
                finetune_run_wrapper(**user_kwargs)
            except Exception as e:
                print(e)
                continue

new_methods = [layerR2_0,layerR1_1]
for method in new_methods:
    for desc_str in desc_str_list1:
        for use_randinit in [True,False]:
            user_kwargs = {k:v for k,v in method.items()}
            user_kwargs['use_rand_init'] = use_randinit
            user_kwargs['desc_str'] = desc_str
            try:
                finetune_run_wrapper(**user_kwargs)
            except Exception as e:
                print(e)
                continue

new_methods = [layerR1_2]
for method in new_methods:
    for desc_str in ['NIVO-OS ADV EVER-OS']:
        for use_randinit in [True,False]:
            user_kwargs = {k:v for k,v in method.items()}
            user_kwargs['use_rand_init'] = use_randinit
            user_kwargs['desc_str'] = desc_str
            try:
                finetune_run_wrapper(**user_kwargs)
            except Exception as e:
                print(e)
                continue            
# main()
# main2()