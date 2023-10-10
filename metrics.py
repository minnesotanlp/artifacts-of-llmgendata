from typing import List
from sklearn.metrics import precision_score

def precision(y_labels: List[int], y_predicted_labels : List[int], average: str):
    '''
    Compute the precision
    y_labels: 1d array-like, or label indicator array
    y_predicteed_labels: 1d array-like, or label indicator array
    average: micro | macro | samples | weighted | binary
    '''
    return precision_score(y_labels, y_predicted_labels, average=average)

def mean_average_precision(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def recall(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def accuracy(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def f1_score(y_labels: List[int], y_predicted_labels : List[int]):
    pass

def auc_score(y_labels: List[int], y_predicted_labels : List[int]):
    pass

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

