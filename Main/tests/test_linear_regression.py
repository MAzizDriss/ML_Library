import numpy as np
import pytest
from models.LinearRegression import utils,model

#testing compute_partials

    #Compute partials fnc lets you compute the derivatives to then update the weights
    #To gain time and reduce complexity numpy's arrays are used to broadcast
    #Reminder that the partial of each weight d(g(w))/dwj = 2*SUM(over i)(h[i:j]*errors[i]) 
    #h[i:j] the weight of the feature j on the ith data point
    #in our case we defined errors = y_hat-y  which is *-1 in the equation
    
def test_compute_partials_simple_example():
    X=np.array([[2,3,4],[5,1,2],[7,1,6]])
    y=np.array([250,300,125])
    y_hat=np.array([200,350,150])
    errors=y_hat - y
    errors=np.reshape(errors,(len(errors),1))
    result=utils.compute_partials(X,errors)
    expected=np.array([[650],[-150],[100]])
    np.testing.assert_array_equal(result,expected)

def test_commpute_partials_when_error_is_null_array():
    X=np.array([[1,1,1],[1,1,1]])
    errors=[0,0]
    errors=np.reshape(errors,(len(errors),1))
    result=utils.compute_partials(X,errors)
    np.testing.assert_array_equal(result,[[0],[0],[0]],"If the erros are 0 which means that the predictions are correct there should be updates for the weighs, therefore the patials = 0 ")

def test_compute_partials_when_dimensions_of_X_is_one_while_errors_diff_of_one():
    X=np.array([1,1,1,1,1]) #5 features #1  data point
    errors=np.array([0,1,2]) #5 data points
    errors=np.reshape(errors,(len(errors),1))
    with pytest.raises(TypeError):
        utils.compute_partials(X,errors)

def test_compute_partials_when_dimensions_of_both_is_one():
    X=np.array([[1,1,1,1,1]]) #5 features #1  data point
    errors=np.array([5]) #1 data points
    errors=np.reshape(errors,(len(errors),1))
    result=utils.compute_partials(X,errors)
    expected=np.array([[10],[10],[10],[10],[10]])
    np.testing.assert_array_equal(result,expected)

def test_compute_partials_when_errors_dimension_is_lower_than_X():
    X=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]) #5 features #3 data points
    errors=np.array([5,2]) #2 data points
    errors=np.reshape(errors,(len(errors),1))
    with pytest.raises(TypeError):
        utils.compute_partials(X,errors)



def test_predict_yhat_simple_example():

    #predict y_hat is a function given weights W: np.array() with the shame of (1,m), b a scalar and a X which is a (n,m) numpy
    #array will return the predicted the value for the n points by multiplying the weights to the feature values
    #for each datapoint and adding the scalar b and returns and n,1 array

    X=np.array(
    [[2,3,4,5,6,7],
    [3,3,3,3,3,3]])
    W=np.array([2,2,2,2,2,2])
    W= W.reshape((len(W),1))
    b=1
    result=utils.predict_yhat(X,W,b)
    expected=np.array([55,37])
    np.testing.assert_array_equal(result,expected)


def test_update_weights_simple_example():
    #W a (1,m) vector 
    #partials a (1,m) vector
    #step_size a scalar
    W=np.array([1,1,1,1],dtype='float').reshape((4,1))
    partials=np.array([-0.5,-0.5,-0.5,-0.5],dtype='float').reshape((4,1))
    step_size=0.1
    result=utils.update_weights(W,partials,step_size)
    expected=np.array([[1.05],[1.05],[1.05],[1.05]])
    np.testing.assert_array_equal(result,expected)

def test_compute_magnitude_basic_example():
    X=np.array([[2,3,4],[5,1,2],[7,1,6]])
    y=np.array([250,300,125])
    y_hat=np.array([200,350,150])
    errors=y_hat - y
    errors=np.reshape(errors,(len(errors),1))
    result=utils.compute_partials(X,errors)
    b_partial=1
    result=utils.compute_magnitude(result,b_partial)
    expected=674.5376
    np.testing.assert_approx_equal(result,expected)


def test_compute_RSS_basic_example():
    y=np.array([1,1,1,1])
    y_hat=np.array([2,2,2,2])
    errors=y_hat-y
    result=utils.compute_RSS(errors)
    expected=4
    np.testing.assert_equal(result,expected)



def test_LR_fit_simple_example():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    initial_weights=np.array([10.,20.])
    reg = model.LinearRegression()
    reg.fit(X,y,initial_weights)
    W,b=reg.get_coef()
    expected_W=[[1],[2]]
    expected_b=3
    np.testing.assert_allclose(W,expected_W,rtol=1e-2)
    np.testing.assert_almost_equal(b,expected_b,decimal=3)



