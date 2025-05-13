import os
import mlflow
from evaluation import find_plots_path

experiment_id_ = None

def find_mlruns_path():
	"""
	Returns:
	--------
	str
		The absolute path to the 'mlruns' directory.
	"""
	current_path = os.getcwd()
	mlruns_path = os.path.join(current_path, '..', 'mlruns')
	return os.path.abspath(mlruns_path)

def mlflow_activation(model):
	"""
	Activates or retrieves an MLflow experiment for tracking the given model.

	The function sets the MLflow tracking URI and creates an experiment with a name 
	based on the model's class. If the experiment already exists, it retrieves its 
	existing ID instead of creating a new one.

	Parameters:
	-----------
	model : object
		A trained machine learning model whose class name will be used to name the experiment.

	Returns:
	--------
	exp_id : str
		The MLflow experiment ID.

	Notes:
	------
	- The experiment name follows the format `{ModelClassName}WithMlflow`.
	- The function relies on `find_mlruns_path()` to locate the MLflow tracking directory.
	"""
	mlflow.set_tracking_uri(find_mlruns_path())
	experiment_name = f'{model.__class__.__name__}WithMlflow'
	try:
		exp_id = mlflow.create_experiment(name=experiment_name)
	except Exception as e:
		exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
	print(f'Experiment {experiment_name} activated.')
	experiment_id_ = exp_id
	return exp_id

def mlflow_logging(exp_id, model, metrics_, artifacts_, X_train, id_column=False, id_name:str=''):
	"""
	Logs model parameters, metrics, artifacts, and the trained model to an MLflow experiment.

	Parameters:
	-----------
	exp_id : str
		The MLflow experiment ID where the data will be logged.

	model : object
		A trained machine learning model that supports `.get_params()` and `.predict()`.

	metrics_ : dict
		Dictionary containing evaluation metrics to be logged.

	artifacts_ : dict
		Dictionary containing visual artifacts (e.g., plots) generated during model evaluation.

	X_train : pandas.DataFrame
		The training dataset used to fit the model.

	id_column : bool, optional (default=False)
		Whether to drop an identifier column before logging the model.

	id_name : str, optional (default='')
		The name of the identifier column to be dropped if `id_column` is True.

	Returns:
	--------
	None

	Notes:
	------
	- Logs hyperparameters retrieved via `model.get_params()`.
	- Logs evaluation metrics stored in `metrics_`.
	- Logs PNG artifacts from the `plots` directory.
	- Logs the trained model with an inferred input signature.
	- The function relies on `find_plots_path()` to locate the artifact directory.
	"""
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