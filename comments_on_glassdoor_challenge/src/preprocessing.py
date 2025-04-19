import pandas as pd
from nltk.corpus import stopwords
#from nltk import stopwords
import os
import re
import string

class dataset():
	def __init__(self):
		pass

	def import_data(self, rows = None, seed = None):
		current_path = os.getcwd()
		data_path = os.path.join(current_path, '..', 'data')
		data_absolute_path = os.path.abspath(data_path)
		files = os.listdir(data_absolute_path)
		for file in files:
			if ('.csv' in file):
				df_name = file
				print(f'Dataset found: ../data/{df_name}')
				break
		df = pd.read_csv(os.path.join(data_absolute_path, df_name))
		if rows is not None:
			df = df.sample(n=rows, random_state = seed).reset_index(drop=True)
		print('Dataset is loaded.')
		return df

	def lower_case_processing(self, series:pd.Series) -> pd.Series:
		return series.apply(lambda x: x.lower() if isinstance(x, str) else x)

	def numbers_remotion(self, series:pd.Series) -> pd.Series:
		return series.apply(lambda x: re.sub(r'\d+', '', x) if isinstance(x, str) else x)

	def punctuation_remotion(self, series:pd.Series) -> pd.Series:
		return series.apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

	def extra_spaces_remotion(self, series:pd.Series) -> pd.Series:
		return series.apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)

	def stopwords_remotion(self, series:pd.Series) -> pd.Series:
		stop_words = set(stopwords.words('english'))
		return series.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

	def clean_column(self, series:pd.Series, lower_case:bool = False, remove_numbers:bool = False, remove_punctuation:bool = False, remove_extra_spaces:bool = False, remove_stopwords:bool = False) -> pd.Series:
		if lower_case:
			series = self.lower_case_processing(series)
			print('Capital letters removed.')
		if remove_numbers:
			series = self.numbers_remotion(series)
			print('Numbers in text removed.')
		if remove_punctuation:
			series = self.punctuation_remotion(series)
			print('Punctuation removed.')
		if remove_extra_spaces:
			series = self.extra_spaces_remotion(series)
			print('Extra spaces removed.')
		if remove_stopwords:
			print(series)
			series = self.stopwords_remotion(series)
			print('Stop words removed.')
			print(series)
		return series

	def text_cleaning(self, df:pd.DataFrame, columns:list, lower_case:bool = False, remove_numbers:bool = False, remove_punctuation:bool = False, remove_extra_spaces:bool = False, remove_stopwords:bool = False) -> pd.DataFrame:
		for column in columns:
			series = self.clean_column(df[column], lower_case=lower_case, remove_numbers=remove_numbers, remove_punctuation=remove_punctuation, remove_extra_spaces=remove_extra_spaces, remove_stopwords=remove_stopwords)
			df[f'{column}_cleaned'] = series
		return df
