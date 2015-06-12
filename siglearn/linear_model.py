# Authors: Jake VanderPlas
# License: BSD

__all__ = ['LinearRegression', 'Ridge', 'Lasso']

import inspect
import types
import imp

import numpy as np
from sklearn import linear_model as sklearn_linear_model

from .base import BaseEstimator


class LinearModel(BaseEstimator):
    """Base class for Linear regression with errors in y"""
    def fit(self, X, y, sigma_y=None):
        X, y = self._construct_X_y(X, y, sigma_y, self.fit_intercept)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self._transform_X(X, self.fit_intercept)
        return self.model.predict(X)

    @staticmethod
    def _transform_X(X, fit_intercept=True):
        X = np.atleast_2d(X)
        if fit_intercept:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        return X

    @staticmethod
    def _construct_X_y(X, y, sigma_y=None, fit_intercept=True):
        """
        Construct the X matrix and y vectors
        scaled appropriately by error in y
        """
        if sigma_y is None:
            sigma_y = 1

        X = np.atleast_2d(X)
        y = np.asarray(y)
        sigma_y = np.asarray(sigma_y)

        # quick sanity checks on inputs. 
        assert X.ndim == 2
        assert y.ndim == 1
        assert sigma_y.ndim in (0, 1, 2)
        assert X.shape[0] == y.shape[0]

        # Intercept is implemented via a column of 1s in the X matrix
        X = LinearModel._transform_X(X, fit_intercept)

        # with no error or constant errors, no scaling needed
        if sigma_y.ndim == 0:
            X_out, y_out = X, y

        elif sigma_y.ndim == 1:
            assert sigma_y.shape == y.shape
            X_out, y_out = X / sigma_y[:, None], y / sigma_y

        elif sigma_y.ndim == 2:
            assert sigma_y.shape == (y.size, y.size)
            evals, evecs = np.linalg.eigh(sigma_y)
            X_out = np.dot(evecs * (evals ** -0.5),
                           np.dot(evecs.T, X))
            y_out = np.dot(evecs * (evals ** -0.5),
                           np.dot(evecs.T, y))
        else:
            raise ValueError("sigma_y must have 0, 1, or 2 dimensions")

        return X_out, y_out


#class LinearRegression(LinearModel):
#    """Ordinary least squares Linear Regression with errors
#    """
#    BaseModel = linear_model.LinearRegression
#
#    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
#        self.fit_intercept = fit_intercept
#        self.normalize = normalize
#        self.copy_X = copy_X
#        self.model = self.BaseModel(fit_intercept=fit_intercept,
#                                    normalize=normalize)


def _model_factory(BaseModel, docstring=None):
    """Generate a siglearn linear model from a scikit-learn linear model"""
    argspec = inspect.getargspec(BaseModel.__init__)
    args = argspec.args
    defaults = argspec.defaults
    
    # Construct __init__() function code
    arglist = ", ".join(arg for arg in args[:len(args) - len(defaults)])
    kwarglist = ", ".join("{0}={1}".format(arg, repr(val))
                          for arg, val in zip(args[len(args) - len(defaults):],
                                              defaults))
    print(kwarglist)
    initcode = "self.model = self.BaseModel({0})".format(', '.join(args[1:]))
    allargs = ", ".join("{arg}={arg}".format(arg=arg)
                        if arg != 'fit_intercept'
                        else 'fit_intercept=False'
                        for arg in args[1:])
    arg_assignments = "\n    ".join("self.{arg} = {arg}".format(arg=arg)
                                    for arg in args[1:])
    initcode = ("def __init__({args}, {kwargs}):\n"
                "    {arg_assignments}\n"
                "    self.model = self.BaseModel({allargs})\n"
                "".format(args=arglist, kwargs=kwarglist, allargs=allargs,
                          arg_assignments=arg_assignments))

    if docstring is None:
        docstring  = BaseModel.__doc__

    # build the class namespace dictionary
    classmembers = dict(__doc__=docstring,
                        BaseModel=BaseModel)
    # TODO: use six._exec here; following is Python 3 only
    exec(initcode, classmembers)

    # return the dynamically-constructed class
    return type(BaseModel.__name__, (LinearModel,), classmembers)

#----------------------------------------------------------------------
# Use the model factory to build some models

LinearRegression = _model_factory(
    sklearn_linear_model.LinearRegression,
    """Ordinary least squares Linear Regression with errors

    TODO: further documentation
    """)
  
Ridge = _model_factory(
    sklearn_linear_model.Ridge,
    """Ridge-regularized Linear Regression with errors

    TODO: further docuementation
    """)       

Lasso = _model_factory(
    sklearn_linear_model.Lasso,
    """Lasso-regularized Linear Regression with errors

    TODO: further documentation
    """)
