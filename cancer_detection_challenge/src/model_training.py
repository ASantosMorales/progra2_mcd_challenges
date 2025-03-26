from sklearn.model_selection import train_test_split

def get_train_test_subsets(X, y, random_state=42, test_size=0.3, shuffle=True):
    """
    Parameters:
    -----------
    X : The feature dataset.
    y : The target series.
    random_state : int, optional, default=42
        The random seed for reproducibility.
    test_size : float, optional, default=0.3
        The proportion of the dataset to include in the test split.
    shuffle : bool, optional, default=True
        Whether to shuffle the data before splitting.

    Returns:
    --------
    X_train, X_test, y_train, y_test : pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series
        The training and testing datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, shuffle=shuffle)
    print(f'Feature subset for training was built. {len(X_train)} samples')
    print(f'Feature subset for testing was built. {len(X_test)} samples')
    print(f'Target subset for training was built. {len(y_train)} samples')
    print(f'Target subset for testing was built. {len(y_test)} samples')
    return X_train, X_test, y_train, y_test

def model_training(model, X_train, y_train, id_column=False, id_name:str=''):
    """
    Parameters:
    X_train : The feature dataset for model training.
    y_train : The target series for model training.

    model : object
        A scikit-learn machine learning model instance that implements the `.fit()` method 
    
    id_column : bool, optional, default=False
        If True, the function will exclude the column specified by `id_name` from X_train before training.
    
    id_name : str, optional, default=''
        The name of the column to exclude if `id_column` is True.
        If `id_column` is True but `id_name` is not in `X_train`, an error will occur.
    
    Returns:
    --------
    model : object
        The trained model instance after fitting on the training data.
    
    Notes:
    ------
    - Ensure that `model` has a `.fit(X, y)` method, as required for most machine learning models.
    """
    if id_column:
        X_train = X_train.drop(columns=[id_name])
    model.fit(X_train, y_train)
    print(f'Model trained.')
    return model