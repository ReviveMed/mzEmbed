from sklearn.base import BaseEstimator
import torch
import numpy as np


# encoder = get_model(encoder_kind, input_size, **encoder_kwargs)
#  encoder_state_dict = torch.load(local_path)
# encoder.load_state_dict(encoder_state_dict)




class PytorchModel(BaseEstimator):

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def fit(self, X, y):
        pass
        # self.model.train()
        # X = torch.tensor(X, dtype=torch.float32)
        # y = torch.tensor(y, dtype=torch.float32)

    def predict(self, X, other_vars=None, head_key=None):
        
        if head_key is None:
            head_key = self.kwargs.get('head_key', None)

        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        if other_vars is None:
            other_vars = torch.zeros(X.shape[0], 1)
        else:
            other_vars = torch.tensor(other_vars, dtype=torch.float32)
            
        head_ouputs = None        
        with torch.inference_mode():
            z = self.model.encoder.transform(X)
            # y = self.model.head(z)
            y_head_output = self.model.head(torch.cat((z, other_vars), 1))

            if isinstance(y_head_output, dict):
                if head_ouputs is None:
                    head_ouputs = {}
                    for k in y_head_output.keys():
                        head_ouputs[k] = y_head_output[k].detach().numpy()
                else:
                    for k in y_head_output.keys():
                        head_ouputs[k] = np.concatenate(
                            (head_ouputs[k], y_head_output[k].detach().numpy()), axis=0)
            else:
                if head_ouputs is None:
                    head_ouputs = y_head_output.detach().numpy()
                else:
                    head_ouputs = np.concatenate(
                        (head_ouputs, y_head_output.detach().numpy()), axis=0)

        if head_key is not None:
            return head_ouputs[head_key]
        else:
            return head_ouputs
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def get_params(self, deep=True):
        return self.kwargs
    

