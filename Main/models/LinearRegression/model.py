from cmath import inf
from models.LinearRegression.utils import gradient_descent

#   The linear regression model should be a class containing methods to fit x and y , give summary of the training process,
#   predict new data, evaluate the model on test data

class LinearRegression():

    def __init__(self,l1=0,l2=0):

        #   The params of linear regression should be the configuration of the model such as lambda 1 and lambda 2
        self.l1=l1
        self.l2=l2
    
    def fit(self,X,y,initial_weights,step_size=0.001,tolerance=0.001,magnitude=inf,max_steps=10000):
        #   The fit function takes input X_train and y_train and run the model then saves the coefficients learnt
        #   The steps to follow are:

        #   Get the number of features m
        m=X.shape[1]

        #   Iniitate weights
        self.W=initial_weights.reshape((m,1))
        self.b=initial_weights[0]

        #   Initiate a list to capture the RSS : useful for ploting later
        RSS=[]

        #   Perform gradient descent algorithm
        self.W,self.b,self.RSS=gradient_descent(X,y,self.W,self.b,self.l1,self.l2,RSS,magnitude,step_size,tolerance,max_steps)
        
    def get_coef(self):

        return self.W,self.b

    def get_RSS(self):
        
        return self.RSS
    
    def evaluate(self,X,y):
        pass

    def predict(self,X):
        pass
