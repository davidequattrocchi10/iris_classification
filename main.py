import pandas as pd  #
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Caricare il dataset Iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

