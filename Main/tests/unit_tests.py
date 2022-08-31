from models.LinearRegression import utils
from models.LinearRegression import model
import numpy as np



def test_predict_yhat():
    X=np.array(
    [[2,3,4,5,6,7],
    [3,3,3,3,3,3]])
    W=np.array([2,2,2,2,2,2])
    W= W.reshape((1,len(W)))
    b=1
    print(utils.predict_yhat(X,W,b))



def test_compute_partials():
    X=np.array([
        [2,3,4],
        [5,1,2],
        [7,1,6]
    ])
    y=np.array([250,300,125])
    y_hat=np.array([200,350,150])
    errors=y - y_hat
    errors=np.reshape(errors,(len(errors),1))
    result=utils.compute_partials(X,errors)
    print(f'X shape: {X.shape} errors shape : {errors.shape}')
    print(f'expected [[650]\n [-150] \n[100]]')
    print(f'got {result} ')



def test_compute_magnitude():
    X=np.array([
        [2,3,4],
        [5,1,2],
        [7,1,6]
    ])
    y=np.array([250,300,125])
    y_hat=np.array([200,350,150])
    errors=y - y_hat
    errors=np.reshape(errors,(len(errors),1))
    result=utils.compute_partials(X,errors)
    print('expected : 674.536')
    print(f'got : {utils.compute_magnitude(result)}')



def test_LR_fit():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    initial_weights=np.array([10.,20.])
    reg = model.LinearRegression()
    W,b,RSS=reg.fit(X,y,initial_weights)
    print(W)
    print(b)

