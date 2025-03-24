import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def find_dataset():
    """
    :return: Absolute path of the first .csv, xlsx or xls file found at ../data/ 
    """
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '..', 'data')
    data_absolute_path = os.path.abspath(data_path)
    files = os.listdir(data_absolute_path)
    for file in files:
        if ('.csv' in file) or ('.xlsx' in file) or ('.xls' in file):
            df_name = file
            print(f'Dataset found: ../data/{df_name}')
            break
    return (os.path.join(data_absolute_path, df_name))

def import_dataset() -> pd.DataFrame:
    """
    :return: Dataframe found at ../data folder.
    """
    df_absolute_path = find_dataset()
    df = pd.read_csv(df_absolute_path)
    print('Dataset is imported')
    return (df)

def dataset_cleaning(df, empty_columns=True, nan_rows=True) -> pd.DataFrame:
    """
    Clean the dataset by removing empty columns and rows with NaN values.
    :param empty_columns: If True, remove columns that are completely empty.
    :param nan_rows: If True, remove rows that have any NaN values.
    :return: Cleaned DataFrame.
    """
    initial_columns_qty = len(df.columns)
    initial_rows_qty = len(df)
    if empty_columns:
        df = df.dropna(axis=1, how='all')
        columns_drop_qty = initial_columns_qty - len(df.columns)
        print(f'Dataset contains {columns_drop_qty} empty columns. These columns have been removed.')
    if nan_rows:
        df = df.dropna(axis=0, how='any')
        rows_drop_qty = initial_rows_qty - len(df)
        if (rows_drop_qty > 0):
            print(f'Dataset contains {rows_drop_qty} rows with NaN values. These rows have been removed.')
    return df

def data_preprocessing(df, target_split=False, target_name:str='', id_column=False, id_name:str='', transform_categorical=False, scaling=False):
    """
    Preprocesses the input DataFrame by handling categorical variables, scaling numerical data, and optionally 
    splitting the dataset into features (X) and target (y).
    
    Parameters:
    ----------
    df : The input DataFrame to preprocess.
    
    target_split : bool, optional, default=False
        If True, splits the DataFrame into features (X) and target (y) based on the column specified by `target_name`.
    
    target_name : str, optional, default=''
        The name of the target column. Used only if `target_split` is True.
    
    id_column : bool, optional, default=False
        If True, excludes the column specified by `id_name` from the scaling process.
    
    id_name : str, optional, default=''
        The name of the column to exclude from scaling if `id_column` is True.
    
    transform_categorical : bool, optional, default=False
        If True, transforms all categorical columns in the DataFrame into numeric using label encoding.
    
    scaling : bool, optional, default=False
        If True, scales the numerical columns in the DataFrame to a range of 0 to 1 using Min-Max scaling.
    
    Returns:
    -------
    pandas.DataFrame or tuple
        - If `target_split=True`, returns a tuple `(X, y)` where `X` contains the feature data and `y` contains the target data.
        - If `target_split=False`, returns the processed DataFrame with optional transformations applied (categorical encoding, scaling).
    """
    if scaling:
        if id_column:
            numeric_cols = df.drop(columns=[id_name]).select_dtypes(include=['float64', 'int64']).columns
        else:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print("Numerical data scaled between 0 and 1.")

    if transform_categorical:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"Categorical column '{col}' transformed into numeric.")
    
    if target_split:
        if target_name not in df.columns:
            raise ValueError(f"Target column '{target_name}' not found in the DataFrame.")
        X = df.drop(columns=[target_name])
        y = df[target_name]
        print(f"Target column '{target_name}' separated.")
        return X, y
    else:
        return df