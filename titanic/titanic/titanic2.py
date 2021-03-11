import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',20)# 设置df最多显示多少列 

train_path='../data/train.csv'
test_path='../data/test.csv'
sub_path='../data/sub.csv'

train=pd.read_csv(train_path)
test=pd.read_csv(test_path)

print(train.head())
#print(train.describe())
#print(train.dtypes)

#print(train.isnull().sum())
print(train.isnull().mean())

# 分类型col和y的关系图
#for col in ['Sex','Pclass','SibSp','Parch']:
#    ax=sns.barplot(x=col,y='Survived',data=train)
#    ax.set_title(col)
#    plt.show()


# Age分段
train['Age']=train['Age'].fillna(-0.5)
test['Age']=test['Age'].fillna(-0.5)
bins=[-1,0,5,12,18,24,35,60,np.inf] # 分段端点
labels=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior'] # 每一段label
train['AgeGroup']=pd.cut(train['Age'],bins,labels=labels)
test['AgeGroup']=pd.cut(test['Age'],bins,labels=labels)

#sns.barplot(x='AgeGroup',y='Survived',data=train)

# 新特征CabinBool
train['CabinBool']=train['Cabin'].notnull().astype('int')
test['CabinBool']=test['Cabin'].notnull().astype('int')

#sns.barplot(x='CabinBool',y='Survived',data=train)
#plt.show()

# 清洗数据
print(test.describe(include='all'))

# 删掉不用的列
train=train.drop(['Cabin','Ticket'],axis=1)
test=test.drop(['Cabin','Ticket'],axis=1)

# 填空值
train=train.fillna({'Embarked':train['Embarked'].mode().iloc[0]})

# 预测Age的空值 通过姓名组的众数填充Age

combine=[train,test]

for dataset in combine:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)

# 交叉表 Index为Title；col为Sex；取值为两者同时出现的次数
print(pd.crosstab(train['Title'],train['Sex']))

# 用提取出的name代替Title
for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady','Capt','Col'\
        'Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title']=dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')

# 输出按照姓名分组的平均存活率
print(train[['Title','Survived']].groupby(['Title'],as_index=False).mean())

# 然后将每一个title组map到数值
title_mapping={'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)



# 获得每个姓名组的众数
mr_age=train[train['Title']==1]['AgeGroup'].mode().iloc[0]# Young Adult
# Q：不加train['AgeGroup']!='Unknown'的话 有可能取值Unkown
miss_age=train[(train['Title']==2) & (train['AgeGroup']!='Unknown')]['AgeGroup'].mode().iloc[0]# Student
mrs_age=train[train['Title']==3]['AgeGroup'].mode().iloc[0]#Adult
master_age=train[train['Title']==4]['AgeGroup'].mode().iloc[0]#Baby
royal_age=train[train['Title']==5]['AgeGroup'].mode().iloc[0]#Adult
rare_age=train[train['Title']==6]['AgeGroup'].mode().iloc[0]#Adult

age_title_mapping={1:mr_age,2:miss_age,3:mrs_age,4:master_age,5:royal_age,6:rare_age}
print(age_title_mapping)

for x in range(len(train['AgeGroup'])):
    if(train['AgeGroup'][x]=='Unknown'):
        train['AgeGroup'][x]=age_title_mapping[train['Title'][x]]

for x in range(len(test['AgeGroup'])):
    if(test['AgeGroup'][x]=='Unknown'):
        test['AgeGroup'][x]=age_title_mapping[test['Title'][x]]

# 将分类特征从str map成1、2、3
age_mapping={'Baby':1,'Child':2,'Teenager':3,'Student':4,'Young Adult':5,'Adult':6,'Senior':7}
train['AgeGroup']=train['AgeGroup'].map(age_mapping)
test['AgeGroup']=test['AgeGroup'].map(age_mapping)

train=train.drop(['Age','Name'],axis=1)
test=test.drop(['Age','Name'],axis=1)

sex_mapping={'male':0,'female':1}
train['Sex']=train['Sex'].map(sex_mapping)
test['Sex']=test['Sex'].map(sex_mapping)

embarked_mapping={'S':1,'C':2,'Q':3}
train['Embarked']=train['Embarked'].map(embarked_mapping)
test['Embarked']=test['Embarked'].map(embarked_mapping)


# 填充test中Fare列：用对应train中pclass的fare平均值
for x in range(len(test['Fare'])):
    if pd.isnull(test['Fare'][x]):
        test['Fare'][x]=round(train[train['Pclass']==test['Pclass'][x]]['Fare'].mean(),4)

# 将数值型col分桶 Q:是train和test各分各的，还是concat再一起分
train['FareBand']=pd.qcut(train['Fare'],4,labels=[1,2,3,4])
test['FareBand']=pd.qcut(test['Fare'],4,labels=[1,2,3,4])

train=train.drop(['Fare'],axis=1)
test=test.drop(['Fare'],axis=1)
#print(test.head())

print(train.isnull().mean())
print(test.isnull().mean())
#### 模型

from sklearn.model_selection import train_test_split

predictors=train.drop(['Survived','PassengerId'],axis=1)
target=train['Survived']

x_train,x_val,y_train,y_val=train_test_split(predictors,target,test_size=0.22,random_state=0)

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

gaussian=GaussianNB()
gaussian.fit(x_train,y_train)
y_pred=gaussian.predict(x_val)
acc_gaussian=round(accuracy_score(y_pred,y_val)*100,2)
print(acc_gaussian)

from sklearn.svm import SVC

svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_val)
acc_svc=round(accuracy_score(y_pred,y_val)*100,2)
print(acc_svc)

from sklearn.tree import DecisionTreeClassifier

decision_tree=DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
y_pred=decision_tree.predict(x_val)
acc_tree=round(accuracy_score(y_pred,y_val)*100,2)
print(acc_tree)

from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier()
random_forest.fit(x_train,y_train)
y_pred=random_forest.predict(x_val)
acc_randomforest=round(accuracy_score(y_pred,y_val)*100,2)
print(acc_randomforest)

