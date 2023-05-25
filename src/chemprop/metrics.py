from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

def pearson(x, y):
    corr, _ = pearsonr(x,y)
    return corr