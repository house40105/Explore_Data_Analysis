"""
@author: House
"""

import pandas as pd # to store tabular data
import numpy as np # to do some math
import matplotlib.pyplot as plt # a popular data visualization tool
import seaborn as sns # another popular data visualization tool
plt.style.use('fivethirtyeight') # a popular data visualization theme
# load in our dataset using pandas

pima = pd.read_csv('pima.data')
pima.head()

pima_column_names = ['times_pregnant', 'plasma_glucose_concentration',
'diastolic_blood_pressure', 'triceps_thickness', 'serum_insulin', 'bmi', 'pedigree_function',
'age', 'onset_diabetes']
pima = pd.read_csv('pima.data', names=pima_column_names)
pima.head()


pima['onset_diabetes'].value_counts(normalize=True)
# get null accuracy, 65% did not develop diabetes
col = 'plasma_glucose_concentration'
plt.hist(pima[pima['onset_diabetes']==0][col], 10, alpha=0.5, label='non-diabetes')
plt.hist(pima[pima['onset_diabetes']==1][col], 10, alpha=0.5, label='diabetes')
plt.legend(loc='upper right')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.title('Histogram of {}'.format(col))

for col in ['bmi', 'diastolic_blood_pressure', 'plasma_glucose_concentration']:
    plt.hist(pima[pima['onset_diabetes']==0][col], 10, alpha=0.5, label='non-diabetes')
    plt.hist(pima[pima['onset_diabetes']==1][col], 10, alpha=0.5, label='diabetes')
    plt.legend(loc='upper right')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(col))
    plt.show()
    

# look at the heatmap of the correlation matrix of our dataset
sns.heatmap(pima.corr())
# plasma_glucose_concentration definitely seems to be an interesting feature here


pima.corr()['onset_diabetes']
# numerical correlation matrix
# plasma_glucose_concentration definitely seems to be an interesting feature here
pima.isnull().sum()
pima.shape # (# rows, # cols)


pima.describe() # get some basic descriptive statistics



pima['serum_insulin'].isnull().sum() # Our number of missing values is (incorrectly) 0
pima['serum_insulin'] = pima['serum_insulin'].map(lambda x:x if x != 0 else None)
# manually replace all 0's with a None value
pima['serum_insulin'].isnull().sum() # check the number of missing values again


# A little faster now for all columns
columns = ['serum_insulin', 'bmi', 'plasma_glucose_concentration', 'diastolic_blood_pressure',
'triceps_thickness']

for col in columns:
    pima[col].replace([0], [None], inplace=True)


pima.isnull().sum() # this makes more sense now!
pima.head()

# doesn't include columns with missing values.
pima.describe()

pima['plasma_glucose_concentration'].mean(), pima['plasma_glucose_concentration'].std()

# drop the rows with missing values
pima_dropped = pima.dropna()
num_rows_lost = round(100*(pima.shape[0] - pima_dropped.shape[0])/float(pima.shape[0]))
print("retained {}% of rows".format(num_rows_lost))
# lost over half of the rows!


# some EDA of the dataset before it was dropped and after
# split of trues and falses before rows dropped
pima['onset_diabetes'].value_counts(normalize=True)
pima_dropped['onset_diabetes'].value_counts(normalize=True)

# the mean values of each column (excluding missing values)
pima.mean()
# the mean values of each column (with missing values rows dropped)
pima_dropped.mean()
# change in means
(pima_dropped.mean() - pima.mean()) / pima.mean()

# change in means as a bar chart
ax = ((pima_dropped.mean() - pima.mean()) / pima.mean()).plot(kind='bar', title='% change in average column values')
ax.set_ylabel('% change')




# now lets do some machine learning
# note we are using the dataset with the dropped rows
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

x_dropped = pima_dropped.drop('onset_diabetes', axis=1)
# create our feature matrix by removing the response variable
print("learning from {} rows".format(x_dropped.shape[0]))
y_dropped = pima_dropped['onset_diabetes'] #response series

# our grid search variables and instances
# KNN parameters to try
knn_params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7]}

knn = KNeighborsClassifier() # instantiate a KNN model
grid = GridSearchCV(knn, knn_params)
grid.fit(x_dropped, y_dropped)
print(grid.best_score_, grid.best_params_)
# but we are learning from way fewer rows…

pima.isnull().sum() # let's fill in the plasma column
empty_plasma_index = pima[pima['plasma_glucose_concentration'].isnull()].index
pima.loc[empty_plasma_index]['plasma_glucose_concentration']

pima['plasma_glucose_concentration'].fillna(pima['plasma_glucose_concentration'].mean(), inplace=True)
# fill the column's missing values with the mean of the rest of the column
pima.isnull().sum() # the column should now have 0 missing values

pima.loc[empty_plasma_index]['plasma_glucose_concentration']

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
pima_imputed = imputer.fit_transform(pima)
type(pima_imputed) # comes out as an array

pima_imputed = pd.DataFrame(pima_imputed, columns=pima_column_names)
# turn our numpy array back into a pandas DataFrame object
pima_imputed.head()
# notice for example the triceps_thickness missing values were replaced with 29.15342

pima_imputed.loc[empty_plasma_index]['plasma_glucose_concentration']
# same values as we obtained with fillna

pima_imputed.isnull().sum() # no missing values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
x_imputed = pima_imputed.drop('onset_diabetes', axis=1)
y_imputed = pima_imputed ['onset_diabetes']
knn_params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, knn_params)
grid.fit(x_imputed, y_imputed)
print(grid.best_score_, grid.best_params_)

pima_zero = pima.fillna(0) # impute values with 0
X_zero = pima_zero.drop('onset_diabetes', axis=1)
y_zero = pima_zero['onset_diabetes']
knn_params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
grid = GridSearchCV(knn, knn_params)
grid.fit(X_zero, y_zero)
print("learning from {} rows".format(X_zero.shape[0]))
print(grid.best_score_, grid.best_params_)
# if the values stayed at 0, our accuracy goes down


from sklearn.model_selection import train_test_split
X = pima[['serum_insulin']].copy()
y = pima['onset_diabetes'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
X.isnull().sum()

entire_data_set_mean = X.mean() # take the entire datasets mean
X = X.fillna(entire_data_set_mean) # and use it to fill in the missing spots
print(entire_data_set_mean)
# Take the split using a random state so that we can examine the same split.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

from sklearn.model_selection import train_test_split
X = pima[['serum_insulin']].copy()
y = pima['onset_diabetes'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

training_mean = X_train.mean()
X_train = X_train.fillna(training_mean)
X_test = X_test.fillna(training_mean)
print(training_mean)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))



from sklearn.pipeline import Pipeline
knn_params = {'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
knn = KNeighborsClassifier()
mean_impute = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('classify', knn)])

X = pima.drop('onset_diabetes', axis=1)
y = pima['onset_diabetes']
grid = GridSearchCV(mean_impute, knn_params)
grid.fit(X, y)
print(grid.best_score_, grid.best_params_)

knn_params = {'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
knn = KNeighborsClassifier()
median_impute = Pipeline([('imputer', SimpleImputer(strategy='median')), ('classify', knn)])
X = pima.drop('onset_diabetes', axis=1)
y = pima['onset_diabetes']
grid = GridSearchCV(median_impute, knn_params)
grid.fit(X, y)
print(grid.best_score_, grid.best_params_)



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
# we will want to fill in missing values to see all 9 columns
pima_imputed_mean = pd.DataFrame(imputer.fit_transform(pima),columns=pima_column_names)
pima_imputed_mean.hist(figsize=(15, 15))

pima_imputed_mean.describe()
pima_imputed_mean.hist(figsize=(15, 15),sharex=True)



# built in z-score normalizer
ax = pima_imputed_mean ['plasma_glucose_concentration'].hist()
ax.set_title('Distribution of plasma_glucose_concentration')

from sklearn.preprocessing import StandardScaler
#glucose_z_score_standardized = scaler.fit_transform(pima_imputed_mean[['plasma_glucose_concentration']])
# note we use the double bracket notation [[ ]] because the transformer requires a dataframe
#ax = pd.Series(glucose_z_score_standardized.reshape(-1,)).hist()
ax.set_title('Distribution of plasma_glucose_concentration after Z Score Scaling')

scale = StandardScaler()
# instantiate a z-scaler object
pima_imputed_mean_scaled = pd.DataFrame(scale.fit_transform(pima_imputed_mean),columns=pima_column_names)
pima_imputed_mean_scaled.hist(figsize=(15, 15), sharex=True)
# now all share the same "space"




from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
knn = KNeighborsClassifier()
knn_params = {'imputer__strategy':['mean', 'median'], 'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
mean_impute_standardize = Pipeline([('imputer', SimpleImputer()), ('standardize', StandardScaler()),
('classify', knn)])
X = pima.drop('onset_diabetes', axis=1)
y = pima['onset_diabetes']
grid = GridSearchCV(mean_impute_standardize, knn_params)
grid.fit(X, y)
print(grid.best_score_, grid.best_params_)


from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
pima_min_maxed = pd.DataFrame(min_max.fit_transform(pima_imputed_mean),columns=pima_column_names)
pima_min_maxed.describe()



