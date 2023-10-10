from typing import List
from sklearn import metrics

def precision(y_labels: List[int], y_predicted_labels : List[int], average: str):
    '''
    Compute the precision
    y_labels: 1d array-like, or label indicator array
    y_predicted_labels: 1d array-like, or label indicator array
    average: micro | macro | samples | weighted | binary
    '''
    return metrics.precision_score(y_labels, y_predicted_labels, average=average)

def mean_average_precision(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def recall(y_labels: List[int], y_predicted_labels : List[int], average: str):
    '''
    Compute the recall
    y_labels: 1d array-like, or label indicator array
    y_predicted_labels: 1d array-like, or label indicator array
    average: micro | macro | samples | weighted | binary
    '''
    return metrics.recall_score(y_labels, y_predicted_labels, average=average)

def accuracy(y_labels: List[int], y_predicted_labels : List[int]):
    '''
    Compute the accuracy
    y_labels: 1d array-like, or label indicator array
    y_predicted_labels: 1d array-like, or label indicator array
    '''
    return metrics.accuracy_score(y_labels, y_predicted_labels)

def f1_score(y_labels: List[int], y_predicted_labels : List[int], average: str):
    '''
    Compute the recall
    y_labels: 1d array-like, or label indicator array
    y_predicted_labels: 1d array-like, or label indicator array
    average: micro | macro | samples | weighted | binary
    '''
    return metrics.f1_score(y_labels, y_predicted_labels, average=average)

def auc_score(y_labels: List[int], y_predicted_labels : List[int | float]):
    '''
    Compute the auc score
    y_labels: 1d array-like, or label indicator array [true labels 0 or 1 for binary classification]
    y_predicted_labels: 1d array-like, or label indicator array [ predicited scores or probabalilties for class 1]
    '''
    return metrics.roc_auc_score(y_labels, y_predicted_labels)

def mean_reciprocal_rank(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def root_mean_squared_error(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def mean_absolute_percentage_error(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def covariance(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def pearson_coefficient(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def spearman_coefficient(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def kendall_tau_coefficient(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def point_biserial_coeffcient(y_labels: List[int], y_predicted_labels : List[int]):
    pass

# print(precision([1,0,0],[1,0,1],'binary'))
# print(recall([1,0,0],[1,0,1],'binary'))
# print(accuracy([1,0,0],[1,0,1]))
# print(f1_score([1,0,0],[1,0,1],'binary'))
# print(auc_score([1,0,0],[1,0,1]))

