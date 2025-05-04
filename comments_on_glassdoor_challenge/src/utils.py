import pandas as pd
import time
import joblib
import os
import numpy as np

def generic_cleanning(series:pd.Series, func:callable) -> pd.Series:
	result = []
	for index, value in series.items():
		try:
			if isinstance(value, str):
				result.append(func(value))
			else:
				result.append(value)
		except Exception as e:
			print(f'Value error "{value}" at index {index}: {e}')
			result.append(value)
	return pd.Series(result, index=series.index)

def timer(func):
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		elapsed_time = end_time - start_time
		print(f'Function "{func.__name__}" took {elapsed_time:4f} s to run.')
		return result
	return wrapper

def store_data(dictionary:dict, data_name:str):
	data_path = os.path.join(os.getcwd(), f'{data_name}.pkl')
	joblib.dump(dictionary, data_path)
	print(f'Data stored at {data_path}.')

def get_data(data_name:str):
	data_path = os.path.join(os.getcwd(), f'{data_name}.pkl')
	bundle = joblib.load(data_path)
	model = bundle['model']
	X_train = bundle['X_train']
	y_train = bundle['y_train']
	X_test = bundle['X_test']
	y_test = bundle['y_test']
	return model, X_train, y_train, X_test, y_test

def store_model(model):
	model_path = os.path.join(os.getcwd(), 'model.pkl')
	joblib.dump(model, model_path)
	print(f'Model saved at {model_path}')

def store_df(df, data_name:str):
	data_path = os.path.join(os.getcwd(), f'{data_name}.csv')
	df.to_csv(data_path, index=False)
	print(f'File {data_name}.csv saved at {data_path}')

def store_sparse_matrix(matrix, matrix_name:str):
	data_path = os.path.join(os.getcwd(), f'{matrix_name}.npz')
	sparse.save_npz(data_path, matrix)
	print(f'File {matrix_name}.npz saved at {data_path}')

def store_array(array, array_name:str):
	array_path = os.path.join(os.getcwd(), f'{array_name}.npy')
	np.save(array_path, array)
	print(f'File {array_name}.npy saved at {array_path}')

"""
	store_model(model=model)
	store_df(df_text_merged, 'df_text_merged')
	store_sparse_matrix(X_test, 'X_test')
	store_array(y_test, 'y_test')

	model = joblib.load(os.path.join(os.getcwd(), 'model.pkl'))
	print(f'Model loaded.')
	df_text_merged = pd.read_csv(os.path.join(os.getcwd(), 'df_text_merged.csv'))
	print(f'DataFrame loaded.')
	X_test = sparse.load_npz(os.path.join(os.getcwd(), 'X_test.npz'))
	print(f'Sparse matrix loaded.')
	y_test = np.load(os.path.join(os.getcwd(), 'y_test.npy'), allow_pickle=True)
	print(f'Array loaded.')
"""