import os

os.system(f'git add mlruns/*')
os.system(f'git commit -m "adding new experiment to run to experiments"')
os.system('git push origin main:features/exp_results')