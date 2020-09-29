import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

pd.set_option('display.max_columns', 23)

df = pd.read_csv('agaricus-lepiota.csv')

df[df.columns] = df[df.columns].apply(lambda col: pd.Categorical(col).codes)

# Selecting the best two features
y = df['edibility']
X = df.drop('edibility', axis=1)

skb = SelectKBest(chi2, k=2)
skb.fit(X, y)
X_new = skb.transform(X)
# With feature selection, we get gill-color and ring-type
print(X.iloc[:, skb.get_support(indices=True)].columns)

# We then run PCA on the same dataset
data_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
pca = PCA(n_components=2)
pca.fit_transform(data_scaled)
maxValues = pd.DataFrame(pca.components_, columns=data_scaled.columns, index=['PC-1', 'PC-2']).max(axis=1)
print(pd.DataFrame(pca.components_, columns=data_scaled.columns, index=['PC-1', 'PC-2']))
print("Max: %s" % maxValues)
# And get spore-print-color and gill-spacing
