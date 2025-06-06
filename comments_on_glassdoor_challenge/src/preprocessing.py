import pandas as pd
from nltk.corpus import stopwords
import spacy
import os
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_dataset(rows: int = None, seed: int = None) -> pd.DataFrame:
    """
    Loads a csv file from the ../data/ directory. If 'rows' is specified,
    returns a random sample of the data.

    Parameters:
    - rows (int, optional): Number of rows to randomly sample.
    - seed (int, optional): Seed for reproducibility.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '..', 'data')
    data_absolute_path = os.path.abspath(data_path)
    files = os.listdir(data_absolute_path)
    df_name = None
    for file in files:
        if file.endswith('.csv'):
            df_name = file
            print(f'Dataset found: ../data/{df_name}')
            break
    if df_name is None:
        raise FileNotFoundError("No CSV file found in ../data")
    df = pd.read_csv(os.path.join(data_absolute_path, df_name))
    if rows is not None:
        df = df.sample(n=rows, random_state=seed).reset_index(drop=True)
    print("Dataset is loaded.")
    return df

def remove_empty_rows(df:pd.DataFrame, list_of_columns:list) -> pd.DataFrame:
	df = df.dropna(subset=list_of_columns)
	print('Rows with empty values were removed from selected columns.')
	return df

def remove_category_from_samples(df:pd.DataFrame, column_name:str, category_label:str) -> pd.DataFrame:
	df = df[df[column_name] != category_label]
	print(f'Category "{category_label}" from column {column_name} was removed.')
	return df

def lower_case_processing(series:pd.Series) -> pd.Series:
	return series.apply(lambda x: x.lower() if isinstance(x, str) else x)

def numbers_remotion(series:pd.Series) -> pd.Series:
	return series.apply(lambda x: re.sub(r'\d+', '', x) if isinstance(x, str) else x)

def punctuation_remotion(series:pd.Series) -> pd.Series:
	return series.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

def extra_spaces_remotion(series:pd.Series) -> pd.Series:
	return series.apply(lambda x: re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else x)

def clean_column(series:pd.Series, lower_case:bool = False, remove_numbers:bool = False, remove_punctuation:bool = False, remove_extra_spaces:bool = False) -> pd.Series:
	if lower_case:
		series = lower_case_processing(series)
		print('Capital letters removed.')
	if remove_numbers:
		series = numbers_remotion(series)
		print('Numbers in text removed.')
	if remove_punctuation:
		series = punctuation_remotion(series)
		print('Punctuation removed.')
	if remove_extra_spaces:
		series = extra_spaces_remotion(series)
		print('Extra spaces removed.')
	return series

def cleanning(df:pd.DataFrame, columns:list, lower_case:bool = False, remove_numbers:bool = False, remove_punctuation:bool = False, remove_extra_spaces:bool = False) -> pd.DataFrame:
	columns_names = []
	for column in columns:
		print(f'Cleaning process for {column}.')
		series = clean_column(df[column], lower_case=lower_case, remove_numbers=remove_numbers, remove_punctuation=remove_punctuation, remove_extra_spaces=remove_extra_spaces)
		columns_names.append(f'{column}_cleaned')
		df[columns_names[-1]] = series
	return df, columns_names

def stopwords_column(series: pd.Series) -> pd.Series:
	stop_words = set(stopwords.words('english'))
	return series.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]) if isinstance(x, str) else x)

def stopwords_remotion(df:pd.DataFrame, columns:list) -> pd.Series:
	columns_names = []
	for column in columns:
		print(f'Stop words remotion process for {column}.')
		series = stopwords_column(df[column])
		columns_names.append(f'{column}_no_stopwords')
		df[columns_names[-1]] = series
		print('Stop words removed.')
	return df, columns_names

def lemmatize_column(series:pd.Series) -> pd.Series:
	return series.apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word) for word in word_tokenize(x)]) if isinstance(x, str) else x)

def lemmatizing(df:pd.DataFrame, columns:list) -> pd.Series:
	columns_names = []
	for column in columns:
		print(f'Lemmatizing process for {column}.')
		series = lemmatize_column(df[column])
		columns_names.append(f'{column}_lemmatized')
		df[columns_names[-1]] = series
		print('Text lemmatized.')
	return df, columns_names

def text_columns_merging(df:pd.DataFrame, columns:list, name_of_column:str = 'text_merged_column') -> pd.DataFrame:
	df[name_of_column] = df[columns].fillna('').agg(' '.join, axis=1)
	print(f'Columns {columns} merged.')
	return df, name_of_column

def rank_mapping(df:pd.DataFrame, columns:list) -> pd.DataFrame:
	value_map = {'v': 2, 'r': 1, 'o': 0, 'x': -2}
	columns_names = []
	for column in columns:
		print(f'Mapping to numeric values for {column} column.')
		columns_names.append(f'{column}_level')
		df[columns_names[-1]] = df[column].map(value_map)
		print('Column mapped.')
	return df, columns_names