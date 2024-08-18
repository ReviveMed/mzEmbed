# %%
from models import create_compound_model_from_info, create_pytorch_model_from_info, MultiHead
import json
import torch
import pandas as pd
input_dir = '/Users/jonaheaton/Downloads'


encoder_info = json.load(open(f'{input_dir}/encoder_info (1).json'))
encoder_state = torch.load(f'{input_dir}/encoder_state_dict (1).pth')

head_info = json.load(open(f'{input_dir}/Cox_OS_info (1).json'))
head_state = torch.load(f'{input_dir}/Cox_OS_state (1).pt')

model = create_compound_model_from_info(encoder_info=encoder_info, 
                                        head_info= head_info,
                                        encoder_state_dict=encoder_state,
                                        head_state_dict=head_state)

model.save_info(input_dir, 'Model_2925 info.json')
model.save_state_to_path(input_dir, 'Model_2925 state.pt')
# %%
model_info = json.load(open(f'{input_dir}/VAE__Cox_OS_info.json'))
model_state = torch.load(f'{input_dir}/VAE__Cox_OS_state.pt')

sklearn_model = create_pytorch_model_from_info(model_info, model_state)


X_data_path = '/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/development_CohortCombination/alignment_RCC_2024_Feb_27/April_30_Finetune_Data/X_finetune_test.csv'
X_data = pd.read_csv(X_data_path,index_col=0)

# %%
preds = sklearn_model.predict(X_data.to_numpy())
# %%
import matplotlib.pyplot as plt

plt.hist(preds)
# %%
import numpy as np
plt.hist(np.log(preds))
# %%
