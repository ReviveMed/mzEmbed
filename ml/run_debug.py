import os
import neptune
import sys

from setup2 import setup_neptune_run

NEPTUNE_API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=='

data_dir = '/DATA'


setup_id = 'pretrain'
run_id = 'RCC-1797'

run_id = setup_neptune_run(data_dir,setup_id=setup_id,with_run_id=run_id,overwrite_existing_kwargs=True)

