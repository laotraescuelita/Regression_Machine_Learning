import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

#Craer datos de simulación.
from sklearn.datasets import make_regression
matriz, vector, coeficientes = make_regression(n_samples = 100,
    n_features = 3,
    n_informative = 3,
    n_targets = 1,
    noise = 0.0,
    coef = True,
    random_state = 1)
print('Matriz \n', matriz[:3])
print('Vector \n', vector[:3])

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(matriz, vector, random_state=0)
# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
print("\nTest set predictions: \n{}".format(reg.predict(X_test)))
print("\nTest set R^2: \n{:.2f}".format(reg.score(X_test, y_test)))


from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(matriz, vector, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print("\nlr.coef_: \n{}".format(lr.coef_))
print("\nlr.intercept_: \n{}".format(lr.intercept_))
print("\nTraining set score: \n{:.2f}".format(lr.score(X_train, y_train)))
print("\nTest set score: \n{:.2f}".format(lr.score(X_test, y_test)))

#Pagina 49
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("\nTraining set score: \n{:.2f}".format(ridge.score(X_train, y_train)))
print("\nTest set score: \n{:.2f}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("\nTraining set score: \n{:.2f}".format(ridge10.score(X_train, y_train)))
print("\nTest set score: \n{:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("\nTraining set score: \n{:.2f}".format(ridge01.score(X_train, y_train)))
print("\nTest set score: \n{:.2f}".format(ridge01.score(X_test, y_test)))


from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))