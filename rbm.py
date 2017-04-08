import numpy as np
import projectLib as lib
import math

# set highest rating
K = 5

def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K))

def sig(x): # CHECKED
    ### TO IMPLEMENT ###
    # x is a real vector of size n
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    ret_i = []
    for element in x:
        ret_i.append(1./(1+math.exp(-element)))
    return np.array(ret_i)

def visibleToHiddenVec(v, w):
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F
    
    h = []
    for j in range(w.shape[1]): 
        sum_x = 0
        for i in range(w.shape[0]): 
            for k in range(5): 
                x = v[i,k]*w[i,j,k]
                sum_x += x 
        h.append(sum_x)
    h_sig = sig(np.array(h)) 
    # print(h_sig)
    return h_sig

def hiddenToVisible(h, w):
    ### TO IMPLEMENT ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.
    
    ### FOR 2-DIMENSIONAL ARRAY... 
    if w.size<=(h.size*5):
        w=np.expand_dims(w,axis=0) #increase dimension of w if the size is F X 5
    
    v = np.array([[0.0 for i in range(5)] for j in range(w.shape[0])])
    sum_num = 0 
    sum_denom = 0 
    sum_denom_inner = 0 
    for i in range(w.shape[0]):
        for k in range(5): 
            for j in range(w.shape[1]): #check if this works, getting F 
                sum_denom_inner += h[j]*w[i,j,k]
            sum_denom += math.exp(sum_denom_inner) 
            sum_denom_inner = 0 
        for k in range(5):
            for j in range(w.shape[1]):
                sum_num += h[j]*w[i,j,k]
            v[i,k] = math.exp(sum_num)/sum_denom
            sum_num = 0 
        sum_denom = 0 
    # print(v)
    return v

def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq):
    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5
    
    h = visibleToHiddenVec(v,w) 
    sample_h = sample(h) 
    p_dist = hiddenToVisible(sample_h, wq)
    # print("p_dist is...")
    # print(p_dist[0])
    
    return p_dist[0]

def predictRatingMax(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of two you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability
    return (ratingDistribution.index(max(ratingDistribution)))+1

def predictRatingExp(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of two you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation of the ratingDistribution
    
    exp_rating = 0 
    for i in range(len(ratingDistribution)): 
        exp_rating+= (i+1)*ratingDistribution[i]
    return exp_rating

def predictMovieForUser(q, user, W, training, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, training, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUser(user, W, training, predictType="exp"):
    ### TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    # print(user,W,training)
    # return None
    # print(user)
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for movie in range(len(W))]
    

# print(sig(np.array([2,3])))
# v = np.array([[0,0,0,1,0],[0,0,1,0,0]])
# #            m1 f1 k--------------k  f2 k---------------k   m2 f1 k-------------k  f2 k---------------k 
# w = np.array([[[0.1,0.2,0.3,0.4,0.5],[0.6,0.7,0.8,0.9,1.0]],[[1.1,1.2,1.3,1.4,1.5],[1.6,1.7,1.8,1.9,2.0]]])
# print(visibleToHiddenVec(v,w))
# # print(sig(np.array([1.7,2.7])))
# # h = visibleToHiddenVec(v,w)
# h=np.array([0.5,0.4])
# print(hiddenToVisible(h,w))