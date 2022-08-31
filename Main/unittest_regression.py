from models.LinearRegression import model, utils
import unittest
import numpy as np


class TestLR(unittest.TestCase):

    def test_compute_partials(self):

        #Compute partials fnc lets you compute the derivatives to then update the weights
        #To gain time and reduce complexity numpy's arrays are used to broadcast
        #Reminder that the partial of each weight d(g(w))/dwj = 2*SUM(over i)(h[i:j]*errors[i]) 
        #h[i:j] the weight of the feature j on the ith data point
        #in our case we defined errors = y_hat-y  which is *-1 in the equation
        #for the defined functions we are not going to use the - sign it is used

        X=np.array([[2,3,4],[5,1,2],[7,1,6]])

        y=np.array([250,300,125])
        y_hat=np.array([200,350,150])

        errors=y_hat - y
        errors=np.reshape(errors,(len(errors),1))
        result=utils.compute_partials(X,errors)
        self.assertEqual(result.tolist(),[[650],[-150],[100]])

        X=np.array([[1,1,1],[1,1,1]])
        errors=[0,0]
        errors=np.reshape(errors,(len(errors),1))
        result=utils.compute_partials(X,errors)
        self.assertEqual(result.tolist(),[[0],[0],[0]])


    

if __name__=="__main__":
    unittest.main()