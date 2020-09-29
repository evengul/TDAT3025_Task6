import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

pd.set_option('display.max_columns', 23)

df = pd.read_csv('agaricus-lepiota.csv')

# This to categorize?
# df[df.columns] = df[df.columns].apply(lambda col: pd.Categorical(col).codes)
# y = df['edibility']
# X = df.drop('edibility', axis=1)

# Or this?
# Selecting the best two features
y = pd.get_dummies(df['edibility'])
X = pd.get_dummies(df.drop('edibility', axis=1))

skb = SelectKBest(chi2, k=5)
skb.fit(X, y)
X_new = skb.transform(X)
# With feature selection, we get
# Index(['bruises?', 'gill-size', 'gill-color', 'stalk-root', 'ring-type'], dtype='object')
print(X.iloc[:, skb.get_support(indices=True)].columns)

# We then run PCA on the same dataset
data_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.columns)
pca = PCA(n_components=5)
pca.fit_transform(data_scaled)
pcaDF = pd.DataFrame(pca.components_, columns=data_scaled.columns)
maxValues = pcaDF.idxmax(axis=1)
print("Max:\n%s" % maxValues)
# And get
# 0    spore-print-color
# 1         gill-spacing
# 2           population
# 3          ring-number
# 4          stalk-shape

# There seems to be no overlap when I do it this way. If I do the get_dummies version, odor
# and stalk-surface-above-ring is in both sets
