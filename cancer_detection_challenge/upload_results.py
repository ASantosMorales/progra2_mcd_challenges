import os
from mlops_pipeline import export_experiment_id

experiment_id = export_experiment_id()

os.system(f'git add mlruns/{experiment_id}*')
os.system(f'git commit -m "adding new experiment to {experiment_id}"')
os.system('git push origin main:features/exp_results')