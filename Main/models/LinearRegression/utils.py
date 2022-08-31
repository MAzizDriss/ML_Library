import numpy as np

def predict_yhat(X,W,b):
    #X is a (n,m) shaped numpy array containing the train data
    #W is a (m,1) shaped array containing the weight values

    #Compute y_hat 
        #matrix multiplication of (n,m) * (m,1) = (n,1)
    y_hat=np.dot(X,W) 
    y_hat += b
    return y_hat.reshape(-1)

def compute_partials(X,errors):
    #X a (n,m) shaped numpy array
    #errors a (1,n) shaped numpy array needs to be transformed into (n,1)
    errors=errors.reshape((len(errors),1) )

    #since we are going to update each weight
    #the formula is :
    #w[j]=W[j]-(etha)*partial[j]
    #partial[j]=derivative of the g(w)/w[j]
    #partial[j]= -2* sum(h[j](xi)*error(xi)) (adjust itself according to the margin of error on each data point)
    # X[:j] is h[j] and we are going to brodecast each h[i,j] * error[i] (each column of the X is multiplied by errors)
    #then sum according to the columns(axis=1)

    #Found a bug due to Tests yeyeyy !! it is due to the nature of broadcasting in numpy 
    #So if given an X as input where len(X)==1 and errors.shape[0]>1 X tend to duplicate its lines to do the multiplication
    
    #Take care of the broadcasting error and any dimentional error
    if X.shape[0] != errors.shape[0]:
        raise TypeError(f'missatching dimensons X:{X.shape} erros:{errors.shape}')

    partials=(X*(2*errors)).sum(axis=0) 
    return partials.reshape((len(partials),1))

def update_weights(W,partials,step_size,l1=0,l2=0):

    #   W a (m,1) vector 
    #   partials a (m,1) vector
    #   step_size a scalar
    #   l2 penalty is lambda 2 for Ridge Regression to resist overfitting to training samples

    W-=step_size* (partials +   2*l2*W  -l1*np.sign(W))

    return W

def compute_magnitude(partials,b_partial):

    #   Partials numpy array shaped (m,1)
    partials.reshape(-1)

    partials=np.append(partials,b_partial)
    magnitude=np.sqrt((partials**2).sum())
    return magnitude

def compute_RSS(errors):
    error_s=errors**2
    RSS =error_s.sum()
    return RSS

def gradient_descent(X,y,W,b,l1,l2,RSS,magnitude,step_size,tolerance,max_steps):
    #   Performs the gradient descent algorihm and return the weights
    i=0
    while(magnitude>tolerance and i<max_steps):
        i+=1
        #   Predict y_hat with the weights (Note the final shape of y_hat is (n,1))
        y_hat=predict_yhat(X,W,b)

        #   calculate the error and RSS
        errors=y_hat-y
        RSS.append(compute_RSS(errors))
            
        #   Updating the weights
        partials=compute_partials(X,errors)
        W= update_weights(W,partials,step_size,l1,l2)
    
        #   Updating b
        b_partial=2*(errors.sum())
        b  =     b   -  step_size*b_partial

        #   Computeing the magnitude
        magnitude=compute_magnitude(partials,b_partial)

    print(f'Number of itterations = {i}')
    return W,b,RSS