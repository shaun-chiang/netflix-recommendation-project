import numpy as np
import projectLib as lib

# shape is movie,user,rating
training = lib.getTrainingData()

#some useful stats
trStats = lib.getUsefulStats(training)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    # ???
    return A

# we also get c
def getc(rBar, ratings):
    # ???
    return None

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    # ???
    return None

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    # ???
    return None

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version
# b = param(A, c)

# Regularised version
l = 1
b = param_reg(A, c, l)

print "Linear regression, l = %f" % l
print lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
