from typing import List
from sklearn import metrics
import statistics

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
    y_predicted_labels: 1d array-like [ predicited scores or probablilties for class 1]
    '''
    return metrics.roc_auc_score(y_labels, y_predicted_labels)

def mean_reciprocal_rank(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def root_mean_squared_error(y_labels: List[int | float], y_predicted_labels : List[int | float]):
    '''
    Compute the root mean squared error
    y_labels: 1d array-like (could be float as well just in case need to use it for regression tasks)
    y_predicted_labels: 1d array-like
    '''
    return metrics.mean_squared_error(y_labels, y_predicted_labels,squared=False)

def mean_absolute_percentage_error(y_labels: List[int | float], y_predicted_labels : List[int | float]):
    '''
    Compute the mean absolute percentage error
    y_labels: 1d array-like (could be float as well just in case need to use it for regression tasks)
    y_predicted_labels: 1d array-like (could be float as well just in case need to use it for regression tasks)
    '''
    return metrics.mean_absolute_percentage_error(y_labels, y_predicted_labels)

def covariance(x: List[int | float], y : List[int | float]):
    '''
    Compute the covariance between two vectors
    x: 1d array-like (could be float as well)
    y: 1d array-like (could be float as well)
    '''
    return statistics.covariance(x, y)

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
# print(root_mean_squared_error([3, -0.5, 2, 7],[2.5, 0.0, 2, 8]))
# print(mean_absolute_percentage_error([3, -0.5, 2, 7],[2.5, 0.0, 2, 8]))
# print(covariance([2, 3, 4, 2],[3, 5, 9, 0]))



