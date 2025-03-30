from sklearn.linear_model import LogisticRegression

from preprocessing import import_dataset, dataset_cleaning, data_preprocessing
from model_training import get_train_test_subsets, model_training
from evaluation import model_predictions, model_evaluation
from mlops_pipeline import mlflow_activation, mlflow_logging

def main():
    df = import_dataset()
    df = dataset_cleaning(df)
    X, y = data_preprocessing(df, 
                              target_split=True, 
                              target_name='diagnosis', 
                              id_column=True, 
                              id_name='id', 
                              transform_categorical=True, 
                              scaling=True)

    X_train, X_test, y_train, y_test = get_train_test_subsets(X, y)

    model_trained = model_training(LogisticRegression(), X_train, y_train, id_column=True, id_name='id')
    y_pred, y_prob = model_predictions(model_trained, X_test, id_column=True, id_name='id', prob=True)
    metrics_, artifacts_ = model_evaluation(y_pred, y_prob, y_test)
    experiment_id = mlflow_activation(model_trained)
    mlflow_logging(exp_id=experiment_id,
                  model=model_trained,
                  metrics_=metrics_,
                  artifacts_=artifacts_,
                  X_train=X_train,
                  id_column=True,
                  id_name='id')

if __name__ == '__main__':
    main()