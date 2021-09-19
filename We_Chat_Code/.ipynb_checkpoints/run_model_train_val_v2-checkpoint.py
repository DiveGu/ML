import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow
import time
import argparse
import os
import json
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from evaluation_v2 import uAUC,compute_weighted_score
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import multiprocessing
cores = multiprocessing.cpu_count()
# print('cpu:{}'.format(cores))

from MyModel import MyModel

SEED=2021
tf.set_random_seed(SEED)

# 存储数据的根目录
ROOT_PATH = "/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data"
TEST_FILE=ROOT_PATH+'/wechat_algo_data1/test_a_concat.csv'
SUB_PATH=ROOT_PATH+'/submit'

ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# ACTION_LIST = ["like"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]

# 设置模型参数
def get_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='mybasemodel')
    
    parser.add_argument('--emb_dim',type=int,default=16)
    parser.add_argument('--word_dim',type=int,default=16)
    parser.add_argument('--dnn_layer',nargs='?',default='[128,128,64]')
    parser.add_argument('--reg_1', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--word_size', type=int, default=150858)
    parser.add_argument('--tag_size', type=int, default=354)
    parser.add_argument('--keyword_size', type=int, default=27272)
    parser.add_argument('--filter_num', type=int, default=4)
    parser.add_argument('--filter_size',nargs='?',default='[2,4,6,8]')
    parser.add_argument('--START_DAY', type=int, default=14)
    parser.add_argument('--LAST_DAY', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()


# 加载训练数据 测试数据
def get_batch_data(df,batch_size,idx):
    start=idx*batch_size
    end=(idx+1)*batch_size
    end=end if end<=df.shape[0] else df.shape[0]
    return df[start:end]

# 把df转为feed_dict
def convert_feed_dict(df,cols,y):
    feed_dict=dict()
    # 将读取的str转为array
    def convert_str_array(x):
        return np.array(json.loads(x))[np.newaxis,:]
    for c in cols:
        feed_dict[c]=df[c].values
        # str转成np.array
        if c in ['des_words','ocr_words','asr_words',\
                 'manual_tag','machine_tag','manual_keywords','machine_keywords']:
            tmp=df[c].apply(lambda x:convert_str_array(x))
            feed_dict[c]=np.concatenate(list(tmp),axis=0) # [N,50]
            
    feed_dict['dnn_keep_prob']=1.0
    if(c in df.columns):
        feed_dict['target']=df[y].values
        feed_dict['dnn_keep_prob']=1.0
    return feed_dict

# 改成并行
def get_batch_feed_dict(df,batch_size,cols,y,idx):
    tmp=get_batch_data(df,batch_size,idx)
    return convert_feed_dict(tmp,cols,y)
    

def convert_feed_list(df,cols,batch_size,y):
    batch_list=[]
    df=df[cols+[y]]
    n_batch=len(df)//batch_size+1
#     pool = multiprocessing.Pool(cores)
    
#     params=[(df,batch_size,cols,y,idx) for idx in range(0,n_batch)]
#     batch_list=pool.map(get_batch_feed_dict,params)
    
     # 将读取的str转为array
    def convert_str_array(x):
        return np.array(json.loads(x))[np.newaxis,:]
    
    for idx in range(n_batch):
        start=idx*batch_size
        end=(idx+1)*batch_size
        end=end if end<=df.shape[0] else df.shape[0]
        batch_df=df[start:end]
        
        feed_dict=dict()
        for c in cols:
            feed_dict[c]=batch_df[c].values
            # str转成np.array
            if c in ['des_words','ocr_words','asr_words',\
                     'manual_tag','machine_tag','manual_keywords','machine_keywords']:
                tmp=batch_df[c].apply(lambda x:convert_str_array(x))
                feed_dict[c]=np.concatenate(list(tmp),axis=0) # [N,50]
            
        feed_dict['dnn_keep_prob']=1.0
        if(c in df.columns):
            feed_dict['target']=batch_df[y].values
            feed_dict['dnn_keep_prob']=1.0
        
        batch_list.append(feed_dict)

    return batch_list


# 读取某个action的sample_conat数据；最后一天为val，其他为train 
def get_df_data(action,day=14):
    df=pd.read_csv('{}/generater_data/{}_{}_concat_sample.csv'.format(ROOT_PATH,action,day))
    return pd.DataFrame(df)

# 制作训练集和验证集 模型输入
def make_train_val(df,day):
    train=df[(df['date_']<day) & (df['date_']>=day-14)]
    day=min(day,14)
    val=df[df['date_']==day]
    return train,val


# 按照batch_size对df进行预测
def predict_by_batch(model,sess,df,batch_size,cols,y):
    predict=[]
    n_batch=df.shape[0]//batch_size+1
    for idx in range(n_batch):
        batch=get_batch_data(df,batch_size,idx)
        values_dict=convert_feed_dict(batch,cols,y)
        feed_dict=model._get_feed_dict(values_dict)
        tmp=model.get_predict(sess,feed_dict)
        predict.append(tmp)
    predict=np.concatenate(predict,axis=0)
    return predict

def main():
    args=get_args()

    START_DAY=args.START_DAY
    LAST_DAY=args.LAST_DAY

    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id',\
                        'video_time_group','feed_cluter',]

    sparse_features = ['userid', 'feedid', 'authorid',]
#     sparse_features+=['user_'+b+'_sum_group' for b in ["read_comment", "like", "click_avatar",  "forward"]]
#     sparse_features+=['user_'+b+'_mean_group' for b in ["read_comment", "like", "click_avatar",  "forward"]]

#     dense_features = ['videoplayseconds','watch_count']+[b+"_sum" for b in ACTION_LIST]
    dense_features = ['videoplayseconds']
    
    text_features=['des_words','ocr_words','asr_words','manual_tag','machine_tag','manual_keywords','machine_keywords']
    text_features=['manual_tag','machine_tag','manual_keywords','machine_keywords']
    
    
    all_text_feature_dict={
                            'des_words':50,
                            'ocr_words':100,
                            'asr_words':100,
                            'manual_tag':4,
                            'machine_tag':6,
                            'manual_keywords':5,
                            'machine_keywords':5,
                           }
    

    # start_day=12 # 相当于取 15-start_day次的val取平均来作为线下指标
    best_auc=dict()
    for action in ACTION_LIST:
        best_auc[action]=[0.0]*(LAST_DAY-START_DAY+1) # day=15时 是全量数据进行train

     #0 读取test原数据
    test=pd.read_csv(TEST_FILE)
    sub_predict=test[['userid', 'feedid']]

    '''
    **************************对于每个action构造模型***********************************
    '''
    for action in ACTION_LIST:
        print('******************{}********************'.format(action))
        epochs=args.epochs
#         sparse_features=sparse_features+['user_'+b+'_sum_group' for b in [action]]+['user_'+b+'_mean_group' for b in [action]]

        # 1 读取train数据集 + 连续特征预处理
        df=get_df_data(action,day=14).sample(frac=1.0)
        if(len(dense_features)>0):
            normal_dense_features=[]
            for c in dense_features:
                if(c in ['videoplayseconds','watch_count']+[b+"_sum" for b in FEA_COLUMN_LIST]):
                    df[c]=np.log(df[c]+1.0)
#                     normal_dense_features.append(c)

            mms = MinMaxScaler(feature_range=(0, 1))
            
            all_dense_concat=df[dense_features].append([test[dense_features]])
            all_dense_concat=mms.fit_transform(all_dense_concat[dense_features])

            df[dense_features] = all_dense_concat[0:len(df),0:len(dense_features)]
            test[dense_features] = all_dense_concat[len(df):len(all_dense_concat),0:len(dense_features)]

        # 构造模型所需要的参数
#         args=get_args()
        sparse_feature_dict=dict()
        for feat in sparse_features:
            sparse_feature_dict[feat]=df[feat].max()+1
        dense_feature_list=dense_features
        
        # 需要用到的文本特征
        text_feature_dict=dict()
        for k in text_features:
            text_feature_dict[k]=all_text_feature_dict[k]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # 2 从start day开始训练到end day
        for day in range(START_DAY,LAST_DAY+1):
            # 2-1 按照day 划分train val
            train,val=make_train_val(df,day)
            userid_list = val['userid'].astype(str).tolist() # val中所有uid列表 计算auc需要使用
            val_labels=val[action].values

            # 2-2 构造模型
            model=MyModel(args,sparse_feature_dict,dense_feature_list,text_feature_dict)
            
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())

            t0=time.time()
            batch_list=convert_feed_list(train,sparse_features+dense_features+text_features,args.batch_size,action)
            t1=time.time()
            print('creat batch list cost:{:.2f}s'.format(t1-t0))
            
            # 2-3 训练epoch
            for epoch in range(epochs):
                t0=time.time()
                loss,task_loss=0.,0.
                train=train.sample(frac=1.0)
                n_batch=train.shape[0]//args.batch_size+1
                for idx in range(n_batch):
                    values_dict=batch_list[idx]
#                     batch_data=get_batch_data(train,args.batch_size,idx)
#                     values_dict=convert_feed_dict(batch_data,sparse_features+dense_features+text_features,action)

                    feed_dict=model._get_feed_dict(values_dict)
                    _,batch_loss,batch_task_loss,batch_reg_loss=model.train(sess,feed_dict)
                    loss+=batch_loss
                    task_loss+=batch_task_loss
        
                loss/=n_batch
                task_loss/=n_batch
                print('epoch:{},loss:{:.5f},task_loss:{:.5f}'.format(epoch+1,loss,task_loss))
                t1=time.time()

                # 2-3-1 在验证集上计算auc
                val_predict_ans=predict_by_batch(model,sess,val,args.batch_size*4,
                                             sparse_features+dense_features+text_features,action)

                t2=time.time()
                auc=uAUC(val_labels, val_predict_ans, userid_list)
                t3=time.time()
                print('start(val) day:{},epoch:{},auc:{},train cost:{:.2f}s,predict cost:{:.2f}s / num:{},auc cost:{:.2f}s'.format(day,epoch+1,auc,t1-t0,t2-t1,val_labels.shape[0],t3-t2))

                 # 2-3-2 更新当前day当前模型的最好auc 
                if(auc>=best_auc[action][day-START_DAY]):
                    best_auc[action][day-START_DAY]=auc
                else:
                    break

    '''
    ********************全部训练完成，计算平均auc和aAuc***********************
    '''
    weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                   "comment": 1, "follow": 1}
    print(best_auc)
    # 更新best_auc取平均 #注：不算day=15的auc
    for action in best_auc.keys():
        tmp=best_auc[action]
        if(LAST_DAY==15):
            best_auc[action]=(sum(tmp)-tmp[-1])/(15-START_DAY)
        else:
            best_auc[action]=(sum(tmp))/(15-START_DAY)
        

    print(best_auc)

    # 保存sub_dict
    weight_auc=compute_weighted_score(best_auc,weight_dict)
    print(weight_auc)


if __name__=='__main__':
    main()
            
