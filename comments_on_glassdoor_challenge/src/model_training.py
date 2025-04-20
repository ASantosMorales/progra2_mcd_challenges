import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class sentimentClassifier():
	def __init__(self):
		pass

	def tfidf_vectorizing(self, series:pd.Series) -> pd.Series:
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(series)
		print('Text vectorized')
		return X

	def split(self, X:pd.Series, y:pd.Series, test_size:float = 0.3):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
		print('Data split for training and testing')
		print(f'X for training = {X_train.shape[0]} samples')
		print(f'X for testing = {X_test.shape[0]} samples')
		print(f'y for training = {y_train.shape[0]} samples')
		print(f'y for testing = {y_test.shape[0]} samples')
		return X_train, X_test, y_train, y_test

	def model_fitting(self, model, X:pd.Series, y:pd.Series):
		model.fit(X, y)
		print(f'Model "{type(model).__name__}" fit performed.')
		return model