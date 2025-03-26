import os
import mlflow
from evaluation import find_plots_path

experiment_id_ = None

def find_mlruns_path():
    current_path = os.getcwd()
    mlruns_path = os.path.join(current_path, '..', 'mlruns')
    return os.path.abspath(mlruns_path)

def mlflow_activation(model):
    mlflow.set_tracking_uri(find_mlruns_path())
    experiment_name = f'{model.__class__.__name__}WithMlflow'
    try:
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(f'Experiment {experiment_name} activated.')
    experiment_id_ = exp_id
    return exp_id

def export_experiment_id():
    exp_id = None
    if experiment_id_ != None:
        exp_id = experiment_id_
    else:
        print('Failure! No experiment id.')
    return exp_id

def mlflow_logging(exp_id, model, metrics_, artifacts_, X_train, id_column=False, id_name:str=''):
    with mlflow.start_run(experiment_id=exp_id):
        # Hyperparameters log
        mlflow.log_param("Model Type", type(model).__name__)
        for hyperparameter, value in model.get_params().items():
            mlflow.log_param(hyperparameter, value)
        print('Hyperparameters logged into mlflow.')
        for metric_ in metrics_:
            mlflow.log_metric(metric_, metrics_[metric_])
        print('Metrics logged into mlflow.')
        for file in os.listdir(find_plots_path()):
            if '.png' in file:
                mlflow.log_artifact(os.path.join(find_plots_path(),file))
                print(f'{file} artifact logged into mlflow.')
        # Log the model itself
        if id_column:
            X_train = X_train.drop(columns=[id_name])
        signature = mlflow.models.signature.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        print('Model logget into mlflow')
        print('Experiment has finished.')

        mlflow.end_run()