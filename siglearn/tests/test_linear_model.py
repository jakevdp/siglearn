import numpy as np
from numpy.testing import assert_allclose

from sklearn import linear_model as sklearn_lm
from .. import linear_model as siglearn_lm

LINEAR_MODELS = siglearn_lm.__all__

def test_all_linear_models(rseed=0, N=30, D=3):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, D)
    y = rng.rand() + np.dot(X, rng.randn(D))
    dy = 1 + rng.rand(N)
    y += dy * rng.randn(N)

    X_scaled = X / dy[:, np.newaxis]
    y_scaled = y / dy

    X_intercept = np.hstack([np.ones((N, 1)), X])
    X_intercept_scaled = X_intercept / dy[:, np.newaxis]

    X_pred = np.random.random((N // 2, D))
    X_pred_intercept = np.hstack([np.ones((N // 2, 1)), X_pred])

    def check_models(model, fit_intercept):
        sk_model = getattr(sklearn_lm, model)(fit_intercept=False)
        if fit_intercept:
            y_sk = sk_model.fit(X_intercept_scaled,
                                y_scaled).predict(X_pred_intercept)
        else:
            y_sk = sk_model.fit(X_scaled, y_scaled).predict(X_pred)
                
        sig_model = getattr(siglearn_lm, model)(fit_intercept=fit_intercept)
        y_sig = sig_model.fit(X, y, dy).predict(X_pred)

        assert_allclose(y_sk, y_sig)

    for model in LINEAR_MODELS:
        for fit_intercept in [True, False]:
            yield check_models, model, fit_intercept

            
        
        
