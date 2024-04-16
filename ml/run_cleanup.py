import neptune
from utils_neptune import get_run_id_list, get_run_id_list_from_query

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='


run_id_list = get_run_id_list(encoder_kind='VAE')


rem_fields = ['both-OS_randinit','both-OS_finetune','both-PFS_randinit','both-PFS_finetune']


for run_id in run_id_list:
    print(run_id)
    run = neptune.init_run(project='revivemed/RCC',
                        api_token=NEPTUNE_API_TOKEN,
                        with_id=run_id,
                        capture_stdout=False,
                        capture_stderr=False,
                        capture_hardware_metrics=False)
    
    run_structure = run.get_structure()

    remov_keys = []
    for key, value in run_structure.items():
        # print(key, value)

        for field in rem_fields:
            if key == field:
                remov_keys.append(key)
                break


    print('deleting the following keys:', remov_keys)
    for key in remov_keys:
        del run[key]
        run.wait()

    run.stop()
    