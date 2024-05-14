from main_finetune import finetune_run_wrapper



method1 = {
    'noise_factor': 0.25,
    'learning_rate': 0.0007869775056037999,
    'l2_reg_weight': 1.0092405183765013e-05,
    'l1_reg_weight': 3.137204254745065e-05,
    'num_epochs': 87,
    'head_hidden_layers': 1,
    'EVER-OS__weight': 10.0,
    'NIVO-OS__weight': 5.0
}

method2 = {
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
    'NIVO-OS__weight': 4.0
}

method3 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.000541422,
    'noise_factor': 0.2,
    'num_epochs': 71,
    'batch_size': 64,
    'head_hidden_layers': 0
}

method4 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.000541422,
    'noise_factor': 0.2,
    'num_epochs': 71,
    'batch_size': 32,
    'head_hidden_layers': 0
}

method5 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.000541422,
    'noise_factor': 0.2,
    'num_epochs': 71,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'remove_nans': True
}

method6 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.0013,
    'noise_factor': 0.2,
    'num_epochs': 98,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'weight_decay': 0.000016,
}

method7 = {
    'dropout_rate': 0.0,
    'encoder_weight': 0.0,
    'learning_rate': 0.0013,
    'noise_factor': 0.2,
    'num_epochs': 98,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'weight_decay': 0.000016,
    'remove_nans': True,
}

method8 = {
    'dropout_rate': 0.4,
    'encoder_weight': 0.75,
    'learning_rate': 0.0021,
    'noise_factor': 0.25,
    'num_epochs': 38,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'weight_decay': 0.0,
    'remove_nans': True,
}

method9 = {
    'dropout_rate': 0.4,
    'encoder_weight': 0.75,
    'learning_rate': 0.0021,
    'noise_factor': 0.25,
    'num_epochs': 38,
    'batch_size': 32,
    'head_hidden_layers': 0,
    'weight_decay': 0.0,
    'remove_nans': True,
}

# user_kwargs = parse_sweep_kwargs_from_command_line()
# method6, method7, method8, method9,method2
for method in  [method3, method4, method5]:
    for desc_str in ['Both-OS','NIVO-OS','EVER-OS','NIVO-OS AND EVER-OS','IMDC','MSKCC','NIVO-OS ADV EVER-OS']:
        for use_randinit in [True,False]:
            user_kwargs = {k:v for k,v in method.items()}
            user_kwargs['use_rand_init'] = use_randinit
            user_kwargs['desc_str'] = desc_str
            try:
                finetune_run_wrapper(user_kwargs)
            except Exception as e:
                print(e)
                continue

# main()
# main2()