import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math

train_path='../data/train.csv'
test_path='../data/test.csv'
sub_path='../data/submission.csv'

# 1 导入数据集
train=pd.read_csv(train_path)
test=pd.read_csv(test_path)

#print(train.head())
#print(test.head())

c_train=train.copy()
c_test=test.copy()

c_train['train']=1
c_test['train']=0
df=pd.concat([c_train,c_test],axis=0,sort=False)

# 2 数据预处理
NAN=[[c,df[c].isna().mean()*100] for c in df.columns]
NAN=pd.DataFrame(NAN,columns=['col_name','percent'])
NAN=NAN.sort_values('percent',ascending=False)
#print(NAN)

NAN=NAN[NAN['percent']>50]

# 先丢弃空值比例太大的列（这里设置为50%）
col_nan=list(NAN['col_name'])
#print(col_nan)

df=df.drop(col_nan,axis=1)
#print(df.head())

# 挑选分类列和数值列
object_columns_df=df.select_dtypes(include=['object'])
numerical_columns_df=df.select_dtypes(exclude=['object'])
#print(object_columns_df.dtypes)
#print(numerical_columns_df.dtypes)

null_counts=object_columns_df.isnull().sum()
print('number of null values in each col\n：{}'.format(null_counts))

columns_fill_none=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',\
    'GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
object_columns_df[columns_fill_none]=object_columns_df[columns_fill_none].fillna('None')

columns_fill_max=['MSZoning','Utilities','Exterior1st','Exterior2nd',\
    'MasVnrType','Electrical','KitchenQual','Functional','SaleType']
object_columns_df[columns_fill_max]=object_columns_df[columns_fill_max].fillna(object_columns_df.mode().iloc[0])

null_counts=numerical_columns_df.isnull().sum()
print('number of null values in each col\n：{}'.format(null_counts))

#print((numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt']).median()) # 35
#print(numerical_columns_df['LotFrontage'].median()) # 68

# GarageYrBlt=YrSold-YearBuilt
numerical_columns_df['GarageYrBlt']=numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold']-35)
numerical_columns_df['LotFrontage']=numerical_columns_df['LotFrontage'].fillna(68)

numerical_columns_df=numerical_columns_df.fillna(0)

#for col in object_columns_df.columns:
#    print(object_columns_df[col].value_counts())
#    object_columns_df[col].value_counts().plot(kind='bar',title=col)
#    plt.show()
    
object_columns_df=object_columns_df.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)

#for col in numerical_columns_df.columns:
#    plt.scatter(range(numerical_columns_df.shape[0]),numerical_columns_df[col])
#    plt.title(col)
#    plt.show()


# 创造新特征

numerical_columns_df['Age_House']=numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt']
print(numerical_columns_df['Age_House'].describe())

# 发现Age_House有负数，进行修改
numerical_columns_df.loc[numerical_columns_df['YrSold']<numerical_columns_df['YearBuilt'],'YrSold']=2009
numerical_columns_df['Age_House']=numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt']

numerical_columns_df['TotalBsmtBath'] = numerical_columns_df['BsmtFullBath'] + numerical_columns_df['BsmtFullBath']*0.5
numerical_columns_df['TotalBath'] = numerical_columns_df['FullBath'] + numerical_columns_df['HalfBath']*0.5 
numerical_columns_df['TotalSA']=numerical_columns_df['TotalBsmtSF'] + numerical_columns_df['1stFlrSF'] + numerical_columns_df['2ndFlrSF']

# 有序特征
bin_map  = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,"None" : 0,
            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6
            }
object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)
object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)
object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)
object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)
object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)
object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)
object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)
object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)
object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)
object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)
object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)
object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)
object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)
object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)

PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}
object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PavedDrive)


print(object_columns_df.dtypes)

# one-hot 特征
rest_object_columns=object_columns_df.select_dtypes(include=['object'])
object_columns_df=pd.get_dummies(object_columns_df,columns=rest_object_columns.columns)

print(object_columns_df.head())

df_final=pd.concat([object_columns_df,numerical_columns_df],axis=1,sort=False)

# 分开train和test
df_final = df_final.drop(['Id',],axis=1)

df_train = df_final[df_final['train'] == 1]
df_train = df_train.drop(['train',],axis=1)

df_test = df_final[df_final['train'] == 0]
df_test = df_test.drop(['SalePrice'],axis=1)
df_test = df_test.drop(['train',],axis=1)

target= df_train['SalePrice']
df_train = df_train.drop(['SalePrice'],axis=1)

# 建模
x_train,x_test,y_train,y_test=train_test_split(df_train,target,test_size=0.33,random_state=0)

xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2400,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=1)


lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=12000, 
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4, 
                                       )

#Fitting
xgb.fit(x_train, y_train)
lgbm.fit(x_train, y_train,eval_metric='rmse')

predict1 = xgb.predict(x_test)
predict = lgbm.predict(x_test)

print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict1))))
print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, predict))))

xgb.fit(df_train, target)
lgbm.fit(df_train, target,eval_metric='rmse')

predict4 = lgbm.predict(df_test)
predict3 = xgb.predict(df_test)
predict_y = ( predict3*0.45 + predict4 * 0.55)

submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": predict_y
    })
submission.to_csv(sub_path, index=False)

