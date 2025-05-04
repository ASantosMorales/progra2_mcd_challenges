import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def tfidf_vectorizing(series:pd.Series) -> pd.Series:
	print('Starting text vectorizing process...')
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(series)
	print('Text vectorized.')
	return X

def get_train_test_subsets(X, y, random_state=42, test_size=0.3, shuffle=True):
	"""
	Splits a dataset into training and testing subsets.

	This function partitions the feature set `X` and target variable `y` into training and testing 
	subsets based on the specified test size, random state, and shuffle preference.

	Parameters:
	-----------
	X : pandas.DataFrame or numpy.ndarray
		The feature dataset.

	y : pandas.Series or numpy.ndarray
		The target variable.

	random_state : int, optional (default=42)
		Controls the shuffling applied to the data before splitting. Ensures reproducibility.

	test_size : float, optional (default=0.3)
		The proportion of the dataset to include in the test split (between 0 and 1).

	shuffle : bool, optional (default=True)
		Whether to shuffle the data before splitting.

	Returns:
	--------
	X_train : pandas.DataFrame or numpy.ndarray
		The training subset of the feature dataset.

	X_test : pandas.DataFrame or numpy.ndarray
		The testing subset of the feature dataset.

	y_train : pandas.Series or numpy.ndarray
		The training subset of the target variable.

	y_test : pandas.Series or numpy.ndarray
		The testing subset of the target variable.

	Notes:
	------
	- The function prints the number of samples in each subset.
	- Uses `train_test_split` from `sklearn.model_selection` for partitioning.
	"""
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, shuffle=shuffle)
	print(f'Feature subset for training was built. {X_train.shape[0]} samples')
	print(f'Feature subset for testing was built. {X_test.shape[0]} samples')
	print(f'Target subset for training was built. {y_train.shape[0]} samples')
	print(f'Target subset for testing was built. {y_test.shape[0]} samples')
	return X_train, X_test, y_train, y_test

def model_training(model, X_train:pd.Series, y_train:pd.Series):
	"""
	Trains a machine learning model using the provided training dataset.

	The function fits the given model to the training data.

	Parameters:
	-----------
	model : object
		A machine learning model that implements the `.fit()` method.

	X_train : pandas.DataFrame
		The training dataset containing feature values.

	y_train : pandas.Series or numpy.ndarray
		The target variable for training.

	Returns:
	--------
	model : object
		The trained model.

	Notes:
	------
	- Prints a confirmation message after training is complete.
	"""
	print(f'Starting training for model "{type(model).__name__}"...')
	model.fit(X_train, y_train)
	print(f'Model "{type(model).__name__}" trained.')
	return model