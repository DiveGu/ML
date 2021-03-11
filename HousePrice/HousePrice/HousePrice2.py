import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


train_path='../data/train.csv'
test_path='../data/test.csv'
sub_path='../data/submission2.csv'

# 1 导入数据集
train=pd.read_csv(train_path)
test=pd.read_csv(test_path)

all_data=pd.concat([train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']])

# 2 数据预处理
#prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
#prices.hist()
#plt.show()

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#numeric_feats为series类型
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index # 数值型列名 list

# 对于数值型特征 计算除去空值后的峰度
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# 类型col进行one-hot
all_data = pd.get_dummies(all_data)

#print(type(all_data.mean()))
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# 3 模型
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
import math

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

# sklearn 模型类
# 3.1 正则化回归
model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

print(cv_ridge.min())

# 3.2 Lasso回归
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print(rmse_cv(model_lasso).mean())

# 这个模型默认有置信度 令不重要的特征系数为0
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# 查看最重要和最不重要的特征
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()

# 3 xgb
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))

preds = 0.7*lasso_preds + 0.3*xgb_preds
print('xgb_rmse'+str(math.sqrt(metrics.mean_squared_error(y, model_xgb.predict(X_train)))))
print('lasso_rmse'+str(math.sqrt(metrics.mean_squared_error(y, model_lasso.predict(X_train)))))
print('xgb+lasso_rmse'+str(math.sqrt(metrics.mean_squared_error(y, 0.7*model_xgb.predict(X_train)+0.3*model_lasso.predict(X_train)))))

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv(sub_path, index = False)
