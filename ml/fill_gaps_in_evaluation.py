
import pandas as pd
import neptune

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

PROJECT_ID = 'revivemed/Survival-RCC'
import pandas as pd

df_file = '/Users/jonaheaton/Desktop/Survival-RCC (4).csv'


df = pd.read_csv(df_file)
print(df.shape)

df[df['desc_str'] == 'both-OS AUX Benefit'].copy()



# df_file = '/Users/jonaheaton/Desktop/Survival-RCC-Filtered.xlsx'

# # df = pd.read_csv(df_file)
# df = pd.read_excel(df_file,header=[0,1])
# df= df['param'].copy()
# # df = df[(df['sweep_kwargs/name'].isin(['layer-R 2.0','layer-R 1.1','layer-R 1.2','basic 2.1']))]
# # df = df[(df['sweep_kwargs/name'].isin(['layer-R 2.0','layer-R 1.1','basic-R 1.0']))]
# # df = df[(df['sweep_kwargs/name'].isin(['layer-R 1.1']))]
# df = df[(df['name'].isin(['layer-R 1.1']))]
# # df = df[(df['desc_str'].str.contains('PFS'))]

# # df = df[~df['desc_str'].str.contains('MSKCC')]
# # df = df[~df['desc_str'].str.contains('IMDC')]

# # df = df[~df['desc str'].str.contains('MSKCC')]
# # df = df[~df['desc str'].str.contains('IMDC')]

print(df.shape)

df.to_csv('/Users/jonaheaton/Desktop/Survival-RCC temp2.csv', index=False)
exit()
# # select the rows that have 'PFS' in the desc_str column
# # df = df[df['desc_str'].str.contains('PFS')]
# finished_ids = []

from main_finetune import finetune_run_wrapper

# for with_id in df['sys/id'].to_list():
# # for with_id in df['Id'].to_list():
# # for with_id in df[('param','sys/id')].to_list():
#     if with_id in finished_ids:
#         continue

#     finetune_run_wrapper(with_id=with_id)
#     finished_ids.append(with_id)

# save the finished_ids to text file
# with open('~/Desktop/finished_ids.txt', 'w') as f:
#     for item in finished_ids:
#         f.write("%s\n" % item)


run_id_list = ['SUR-2847','SUR-2848']
for with_id in run_id_list:
    finetune_run_wrapper(with_id=with_id)
