
import neptune
NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='
from utils_neptune import count_fields
run_id = 'RCC-1915'
from neptune.exceptions import NeptuneException

run = neptune.init_run(project='revivemed/RCC',
        api_token=NEPTUNE_API_TOKEN,
        with_id=run_id,
        capture_stdout=False,
        capture_stderr=False,
        capture_hardware_metrics=False)

num_fields = count_fields(run.get_structure())
print(num_fields)


# del run['finetune_IMDCa']
# del run['randinit_IMDCa']
for ii in [20]:
    try:
        del run[f'finetune_optuna_IMDC__{ii}']
        del run[f'randinit_optuna_IMDC__{ii}']
    except NeptuneException:
        print('not found')

run.stop()