import numpy as np
import projectLib as lib

# shape is movie,user,rating
training = lib.getTrainingData()

#some useful stats
trStats = lib.getUsefulStats(training)
rBar = np.mean(trStats["ratings"])

def genR():
    # We transform the training data into R, like in our homework.
    # We first get the dimensions of the array.
    M=-1
    U=-1
    for row in training:
        if row[0]>M:
            M=row[0]
        if row[1]>U:
            U=row[1]
    R = [[0 for i in range(M+1)] for j in range(U+1)]
    # Now we place all the ratings in the array.
    for row in training:
        R[row[1]][row[0]]=row[2]
        
    return np.array(R)

# we get the A matrix from the training dataset
def getA(R):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    current_input = 0
    for i, row in enumerate(R):
        for j, element in enumerate(row):
            if element != 0:
                A[current_input][i] = 1
                A[current_input][j + 5] = 1
                current_input += 1
    return A

# we also get c0
def getc(rBar, ratings):
    C = []
    for element in ratings:
        C.append([element - rBar])
    return np.array(C)

R = genR()
# apply the functions
A = getA(R)
c = getc(rBar, trStats["ratings"])
# print(R)
# print(A)
# print(c)


# compute the estimator b
def param(A, c):
    aTa = np.dot(A.T, A) 
    aTc = np.dot(A.T,c) 
    B = np.linalg.lstsq(aTa,aTc)
    return B[0]

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    aTa = np.dot(A.T, A) 
    aTc = np.dot(A.T,c) 
    B = np.linalg.lstsq(aTa+l*np.identity(A.shape[1]),aTc)
    return B[0]


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
b = param(A, c)
print("Linear regression, without regularization")
print(lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))

# Regularised version
l = 1
b = param_reg(A, c, l)
print("Linear regression, l = %f" % l)
print(lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
