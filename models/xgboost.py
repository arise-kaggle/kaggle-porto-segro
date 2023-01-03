import xgboost as xgb
import numpy as np
import logging

from logs.logger import log_evaluation

def eval_gini(y_true, y_pred):
    # Verify that the actual and predicted values are the same size (different values raise errors)
    assert y_true.shape == y_pred.shape

    n_samples = y_true.shape[0] # Number of data
    L_mid = np.linspace(1 / n_samples, 1, n_samples) # Diagonal value

    # 1) Gini coefficient for predicted values
    pred_order = y_true[y_pred.argsort()] # Sort y_true values by y_pred size
    L_pred = np.cumsum(pred_order) / np.sum(pred_order) # Lorentz Curve
    G_pred = np.sum(L_mid - L_pred) # Gini coefficient for predicted values

    # 2) Gini coefficient when prediction is perfect
    true_order = y_true[y_true.argsort()] # Sort y_true values by y_true size
    L_true = np.cumsum(true_order) / np.sum(true_order) # Lorentz Curve
    G_true = np.sum(L_mid - L_true) #  Gini coefficient when prediction is perfect

    # Normalized Gini coefficient
    return G_pred / G_true

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', eval_gini(labels, preds)

def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params):

    # データセットを生成する
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    dtest = xgb.DMatrix(X_test)

    logging.debug(params)

    # ロガーの作成
    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=30)]

    # 上記のパラメータでモデルを学習する
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model = xgb.train(params=params, 
                           dtrain=dtrain,
                           num_boost_round=1000,
                           evals=watchlist,
                          maximize=True,
                           feval=gini_xgb,
                           early_stopping_rounds=150,
                           verbose_eval=100)


    # テストデータを予測する
    y_pred = model.predict(dtest, ntree_limit=model.best_iteration)

    return y_pred, model
