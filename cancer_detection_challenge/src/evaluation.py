from sklearn import metrics
import os
import matplotlib.pyplot as plt

def roc_curve_generation(y_test, y_prob, auc_metric):
    """
    Generates the ROC curve for a given classification model.

    Parameters:
    -----------
    y_test : array-like
        Ground truth binary labels (0 or 1).
    
    y_prob : array-like
        Predicted probabilities for the positive class.
    
    auc_metric : float
        The computed AUC value for the ROC curve.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the ROC curve plot.
    """
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    # ROC curve plotting
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, linestyle=':', marker='o', color='red', linewidth=1, markersize=4, label=f'AUC = {round(auc_metric, 4)}')
    ax.plot([0, 1], [0, 1], linestyle=':', color='gray')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR = Sensitivity)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc='lower right')
    fig.show()
    return fig

def find_plots_path():
    """
    Returns:
    --------
    str
        The absolute path to the 'plots' directory.
    """
    current_path = os.getcwd()
    plots_path = os.path.join(current_path, '..', 'plots')
    return os.path.abspath(plots_path)

def model_predictions(model, X_test, id_column=False, id_name:str='', prob=False):
    """
    Generates predictions from a trained model on a given test dataset.

    Parameters:
    -----------
    model : object
        A trained machine learning model that supports `.predict()` and optionally `.predict_proba()`.
    
    X_test : pandas.DataFrame
        The test dataset containing feature values.

    id_column : bool, optional (default=False)
        Whether to drop an identifier column before making predictions.

    id_name : str, optional (default='')
        The name of the identifier column to be dropped if `id_column` is True.

    prob : bool, optional (default=False)
        Whether to return probability predictions in addition to class predictions.

    Returns:
    --------
    numpy.ndarray
        Predicted class labels.
    
    tuple of numpy.ndarray, optional
        If `prob=True`, returns a tuple containing:
        - `y_pred`: Predicted class labels.
        - `y_prob`: Predicted probabilities for the positive class.
    """
    if id_column:
        X_test = X_test.drop(columns=[id_name])
    y_pred = model.predict(X_test)
    print('Model predictions done.')
    if prob:
        y_prob = model.predict_proba(X_test)[:,1]
        print('Model probabilistic predictions done.')
        return y_pred, y_prob
    else:
        return y_pred
    
def model_evaluation(y_pred, y_prob, y_test, acc=True, f1=True, conf_mat=True, auc=True, roc=True):
    """
    Evaluates a classification model using multiple performance metrics and visualizations.

    The function computes accuracy, F1-score, AUC score, and generates a confusion matrix 
    and ROC curve based on the given predictions and ground truth labels. The computed metrics 
    are stored in a dictionary, and generated plots are saved in the 'plots' directory.

    Parameters:
    -----------
    y_pred : array-like
        Predicted class labels.

    y_prob : array-like
        Predicted probabilities for the positive class.

    y_test : array-like
        Ground truth binary labels (0 or 1).

    acc : bool, optional (default=True)
        Whether to compute and display the accuracy score.

    f1 : bool, optional (default=True)
        Whether to compute and display the F1-score.

    conf_mat : bool, optional (default=True)
        Whether to compute and plot the confusion matrix.

    auc : bool, optional (default=True)
        Whether to compute and display the AUC score.

    roc : bool, optional (default=True)
        Whether to generate and save the ROC curve (requires `auc=True`).

    Returns:
    --------
    metrics_ : dict
        A dictionary containing the computed evaluation metrics (accuracy, F1-score, AUC score).

    artifacts_ : dict
        A dictionary containing generated visualizations:
        - 'confusion_matrix': ConfusionMatrixDisplay object (if `conf_mat=True`).
        - 'roc': Matplotlib figure of the ROC curve (if `roc=True`).

    Notes:
    ------
    - Plots (confusion matrix and ROC curve) are saved in the 'plots' directory.
    - The function relies on `find_plots_path()` for locating the directory.
    """
    metrics_ = {}
    artifacts_ = {}
    if acc:
        metrics_['accurracy'] = metrics.accuracy_score(y_test, y_pred)
        print(f'Model accurracy: {round(metrics_['accurracy'] * 100, 2)}%')
    if f1:
        metrics_['f1_score'] = metrics.f1_score(y_test, y_pred)
        print(f'Model F1 score: {round(metrics_['f1_score'], 4)}')
    if conf_mat:
        cm = metrics.confusion_matrix(y_test, y_pred)
        cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_disp.plot().figure_.savefig(os.path.join(find_plots_path(),'confusion_matrix.png'))
        artifacts_['confusion_matrix'] = cm_disp
        print('Confusion matrix plotted.')
    if auc:
        metrics_['auc_score'] = metrics.roc_auc_score(y_test, y_prob)
        print(f'Model AUC score: {round(metrics_['auc_score'], 4)}')
        if roc:
            roc_fig = roc_curve_generation(y_test, y_prob, metrics_['auc_score'])
            artifacts_['roc'] = roc_fig
            roc_fig.savefig(os.path.join(find_plots_path(),'roc.png'))
            print('ROC plotted.')
    return metrics_, artifacts_