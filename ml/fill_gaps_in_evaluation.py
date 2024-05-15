
import pandas as pd
import neptune

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'
import pandas as pd

df_file = '/Users/jonaheaton/Desktop/Survival-RCC (4).csv'
df = pd.read_csv(df_file)


from main_finetune import finetune_run_wrapper

finished_ids = ['SUR-2815',
 'SUR-2814',
 'SUR-2813',
 'SUR-2812',
 'SUR-2811',
 'SUR-2810',
 'SUR-2809']
# 'SUR-2808'

for with_id in df['Id'].to_list():
# for with_id in df[('param','sys/id')].to_list():
    if with_id in finished_ids:
        continue
    # if with_id == 'SUR-2808':
        # print('Skipping SUR-2808')
        # continue
    finetune_run_wrapper(with_id=with_id)
    finished_ids.append(with_id)