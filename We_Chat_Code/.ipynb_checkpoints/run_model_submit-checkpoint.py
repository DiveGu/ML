import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow
import time
import argparse
import random
import os
import json
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from evaluation_v2 import uAUC,compute_weighted_score
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from MyModel import MyModel
SEED=2021
tf.set_random_seed(SEED)

# 存储数据的根目录
ROOT_PATH = "/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data"
DATASET_PATH=ROOT_PATH+'/wechat_algo_data1'
TEST_FILE=ROOT_PATH+'/wechat_algo_data1/test_b.csv'
SUB_PATH=ROOT_PATH+'/submit'

ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 15, "like": 15, "click_avatar": 10, "forward": 10, "comment": 10, "follow": 10, "favorite": 10}
# 各个行为构造训练数据的天数
ACTION_DAY_NUM = {"read_comment": 14, "like": 14, "click_avatar": 14, "forward": 14, "comment": 14, "follow": 14, "favorite": 14}

lr_list=[3e-4,4e-4,5e-4,6e-4,7e-4,8e-4]
seed_list=[2021,1997,6666,6677,1317]
# 设置模型参数
def get_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='wdl_des_tag_keyword_sample')
    
    parser.add_argument('--emb_dim',type=int,default=16)
    parser.add_argument('--word_dim',type=int,default=16)
    parser.add_argument('--dnn_layer',nargs='?',default='[128,128,64]')
    parser.add_argument('--reg_1', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=5e-4)
    #parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--word_size', type=int, default=150858)
    parser.add_argument('--tag_size', type=int, default=354)
    parser.add_argument('--keyword_size', type=int, default=27272)
    parser.add_argument('--filter_num', type=int, default=12)
    parser.add_argument('--filter_size',nargs='?',default='[1,2,4,6,8,12]')
    parser.add_argument('--save_flag', type=bool, default=False)
    parser.add_argument('--load_flag', type=bool, default=False)
    parser.add_argument('--repeat', type=int, default=5)

    return parser.parse_args()

sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id',\
                        'video_time_group','feed_cluter',]
    
dense_features = ['videoplayseconds']
    

text_features=['des_words','manual_tag','machine_tag','manual_keywords','machine_keywords']

# 创造path目录
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)       
    else:
        return

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
                 'manual_tag','machine_tag','manual_keywords','machine_keywords',\
                'user_like_history',"user_read_comment_history",\
                    "user_click_avatar_history",  "user_forward_history"]:
            tmp=df[c].apply(lambda x:convert_str_array(x))
            feed_dict[c]=np.concatenate(list(tmp),axis=0) # [N,50]
            if(c in ['user_like_history',"user_read_comment_history",\
                    "user_click_avatar_history",  "user_forward_history"]):
                feed_dict[c]=feed_dict[c][:,0:3]
            
            
    feed_dict['dnn_keep_prob']=1.0
    if(y in df.columns):
        feed_dict['target']=df[y].values
        feed_dict['dnn_keep_prob']=1.0
    return feed_dict

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
                 'manual_tag','machine_tag','manual_keywords','machine_keywords',\
                    'user_like_history',"user_read_comment_history",\
                    "user_click_avatar_history",  "user_forward_history"]:
                tmp=batch_df[c].apply(lambda x:convert_str_array(x))
                feed_dict[c]=np.concatenate(list(tmp),axis=0) # [N,50]
                if(c in ['user_like_history',"user_read_comment_history",\
                    "user_click_avatar_history",  "user_forward_history"]):
                    feed_dict[c]=feed_dict[c][:,0:3]
            
        feed_dict['dnn_keep_prob']=1.0
        if(c in df.columns):
            feed_dict['target']=batch_df[y].values
            feed_dict['dnn_keep_prob']=1.0
        
        batch_list.append(feed_dict)

    return batch_list

# 读取某个action的sample_conat数据；最后一天为val，其他为train 
def get_df_data(action,day=14):
    df=pd.read_csv('{}/generater_data/{}_{}_sample.csv'.format(ROOT_PATH,action,day))
    return pd.DataFrame(df)

# 制作训练集和验证集 模型输入
def make_train_val(df,day):
    train=df[(df['date_']<day) & (df['date_']>=day-15)]
    day=min(day,14)
    val=df[df['date_']==day]
    return train,val

# 采样History数据
def generate_sample(df,action,day=14):
    """
    对负样本进行下采样，生成各个阶段所需样本
    """
    df = df.drop_duplicates(subset=['userid', 'feedid', action], keep='last')
    
    # 负样本下采样
    action_df = df[(df["date_"] <= day) & (df["date_"] >= day - ACTION_DAY_NUM[action] + 1)]
    df_neg = action_df[action_df[action] == 0]
                 
    all_pos_num=len(action_df[action_df[action] == 1])
    all_neg_num=len(action_df)-all_pos_num
    
#     if(action in ['read_comment','like',"click_avatar",  "forward"]):
#         design_action=['read_comment','like','follow','favorite','forward','comment',"click_avatar",]
#         design_action.remove(action)
#         df_neg_design=df_neg[(df_neg[design_action[0]]==1) | (df_neg[design_action[1]]==1) | (df_neg[design_action[2]]==1) |
#                             (df_neg[design_action[3]]==1) | (df_neg[design_action[4]]==1)| (df_neg[design_action[5]]==1)]
#         df_neg=df_neg[~df_neg.index.isin(df_neg_design.index)]
        
    sample_neg_num=min(len(df_neg),all_pos_num*ACTION_SAMPLE_RATE[action])
    df_neg=df_neg.sample(n=sample_neg_num, random_state=SEED, replace=False)
    print('-----------{}-------------'.format(action))
    print('pos num:{};neg num:{}'.format(all_pos_num,sample_neg_num))
#   #每个aciton进行负采样
# #   #按照停留时间进行采样
#     df_neg=df_neg.sort_values(by='play',ascending=False)
#     df_neg = df_neg[:sample_neg_num]
    
    df_all = pd.concat([df_neg,action_df[action_df[action] == 1]])  
    col = ["userid", "feedid", "date_", "device"] + ACTION_LIST
    
    return df_all[col]


# 把train拼接上 u i特征
def train_concat(sample,action):
    # 用户基本特征
    df_users=pd.read_csv(DATASET_PATH+'/user_info.csv')
    df_users = df_users.set_index('userid')
    # 用户统计特征
    df_users_static=pd.read_csv(DATASET_PATH+'/user_feature_sum_avg.csv')
    df_users_static=df_users_static.drop_duplicates(subset=['userid','date_'], keep='last')
    df_users_static=df_users_static.set_index(['userid','date_']) # 必须重新设置idx 不然join的时候报错
    # 视频特征
    df_feed=pd.read_csv(DATASET_PATH+'/feed_feature.csv')
    df_feed = df_feed.set_index('feedid')
    
    features = ["userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id",\
                'watch_count_group','video_time_group','feed_cluter',\
                "videoplayseconds","watch_count","play_times",'date_','des_words','ocr_words','asr_words',\
                'manual_tag','machine_tag','manual_keywords','machine_keywords','feed_emb_id']
    features=features+['user_'+b+'_sum_group' for b in FEA_COLUMN_LIST]+['user_'+b+'_mean_group' for b in FEA_COLUMN_LIST]
        
    sample = sample.join(df_feed, on="feedid", how="left", rsuffix="_feed")
    sample = sample.join(df_users, on=["userid"], how="left", rsuffix="_user_id")
    sample = sample.join(df_users_static, on=["userid", "date_"], how="left", rsuffix="_user_static")
        
    # 把各种统计信息更新到features中
    user_feature_col = [b+"_sum" for b in FEA_COLUMN_LIST]+[b+"_mean" for b in FEA_COLUMN_LIST]
    sample[user_feature_col] = sample[user_feature_col].fillna(0.0)
        
    features += user_feature_col
    features+=[action]
    
    # id=0 填充未知分类数据和离散数据
    sample[["authorid", "bgm_song_id", "bgm_singer_id",'watch_count_group','video_time_group']] += 1  
    sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds",'watch_count_group','video_time_group']] = \
        sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds",\
                'watch_count_group','video_time_group']].fillna(0)
        
    # 给数值型数据增加非线性
    dense_cols=['videoplayseconds','watch_count']+user_feature_col

    # 把分类数据id转化成int格式
    sample[["authorid", "bgm_song_id", "bgm_singer_id",'watch_count_group','video_time_group']] = \
    sample[["authorid", "bgm_song_id", "bgm_singer_id",'watch_count_group','video_time_group']].astype(int)
        
    return sample[features]


# 把test数据 拼接上 u i特征
def test_concat(df_test):
    # 用户基本特征
    df_users=pd.read_csv(DATASET_PATH+'/user_info.csv')
    df_users = df_users.set_index('userid')
    # 用户统计特征
    df_users_static=pd.read_csv(DATASET_PATH+'/user_feature_sum_avg.csv')
    df_users_static=df_users_static.drop_duplicates(subset=['userid','date_'], keep='last')
    # test的时候直接使用14天的统计数据
    df_users_static=df_users_static[df_users_static['date_']==14]
    df_users_static=df_users_static.set_index('userid')
    
    # 视频特征
    df_feed=pd.read_csv(DATASET_PATH+'/feed_feature.csv')
    df_feed = df_feed.set_index('feedid')
    
    features = ["userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id",\
                'watch_count_group','video_time_group','feed_cluter',\
                "videoplayseconds","watch_count","play_times",'des_words','ocr_words','asr_words',\
                'manual_tag','machine_tag','manual_keywords','machine_keywords','feed_emb_id']
    
    features=features+['user_'+b+'_sum_group' for b in FEA_COLUMN_LIST]+['user_'+b+'_mean_group' for b in FEA_COLUMN_LIST]

    sample=df_test
    sample = sample.join(df_feed, on="feedid", how="left", rsuffix="_feed")
    sample = sample.join(df_users, on="userid", how="left", rsuffix="_user_id")
    sample = sample.join(df_users_static, on="userid", how="left", rsuffix="_user_static")

    # 把各种统计信息更新到features中
    user_feature_col = [b+"_sum" for b in FEA_COLUMN_LIST]+[b+"_mean" for b in FEA_COLUMN_LIST]
    # test中可能有冷启动 所以必须填充空值
    sample[user_feature_col] = sample[user_feature_col].fillna(0.0)

    features += user_feature_col

    # id=0 填充未知分类数据和离散数据
    sample[["authorid", "bgm_song_id", "bgm_singer_id",'watch_count_group','video_time_group']] += 1  
    sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds",'watch_count_group','video_time_group']] = \
        sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds",\
                'watch_count_group','video_time_group']].fillna(0)

    dense_cols=['videoplayseconds','watch_count']+user_feature_col

    # 把分类数据id转化成int格式
    sample[["authorid", "bgm_song_id", "bgm_singer_id",'watch_count_group','video_time_group']] = \
        sample[["authorid", "bgm_song_id", "bgm_singer_id",'watch_count_group','video_time_group']].astype(int)

    return sample[features]

# merge history list
def concat_history(df1,df2,key):
    ret=pd.merge(df1,df2,how='left',on=key)
    # 注意：相同列名变成了_x _y
    ret=ret.rename({'date__x':'date_'},axis='columns')
    # 注意：使用join没有_x _y 但是要set_index
#     df1.join(df2,on=key, how="left", rsuffix='tmp')
    return ret



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

# 根据path加载
def load_pretrain(load_flag,path):
    if(load_flag):
        try:
            pretrain_data=np.load(path)
            return pretrain_data
        except:
            return None
    else:
        return None

def main():
    args=get_args()
    predict_best=dict()
    
    
    all_text_feature_dict={
                            'des_words':50,
                            'ocr_words':100,
                            'asr_words':100,
                            'manual_tag':4,
                            'machine_tag':6,
                            'manual_keywords':5,
                            'machine_keywords':5,
                            'user_like_history':3,
                            "user_read_comment_history":3, 
                            "user_click_avatar_history":3,
                            "user_forward_history":3,
                           }
    
    # 每个action最佳的训练epochs lr=1e-4
    action_epochs={"read_comment":[2,3],
               "like":[1,1,2],
               "click_avatar":[1],
               "forward":[1,2],}
    # lr>1e-4之后
    action_epochs={"read_comment":[1],
               "like":[1],
               "click_avatar":[1],
               "forward":[1],}


    #0 读取test原数据
    df_history_list=pd.read_csv(ROOT_PATH+'/wechat_algo_data1/user_history_list.csv')
    test=pd.read_csv(TEST_FILE)
    test=test_concat(test)
    test['date_']=15
    test=concat_history(test,df_history_list,['userid','date_'])
    sub_predict=test[['userid', 'feedid']]
    
    df_actions=pd.read_csv(DATASET_PATH+'/user_action.csv')
    '''
    **************************对于每个action构造模型***********************************
    '''
    for action in ACTION_LIST:
        print('******************{}********************'.format(action))

        # 1 读取train数据集 + 连续特征预处理
#         df=get_df_data(action,day=14).sample(frac=1.0)
        df=generate_sample(df_actions,action,day=14)
        df=train_concat(df,action)
        df=concat_history(df,df_history_list,['userid','date_'])
        for c in dense_features:
            df[c]=np.log(df[c]+1.0)
        
        mms = MinMaxScaler(feature_range=(0, 1))
    
        all_dense_concat=df[dense_features].append([test[dense_features]])
        all_dense_concat=mms.fit_transform(all_dense_concat[dense_features])
    
        df[dense_features] = all_dense_concat[0:len(df),0:len(dense_features)]
        test[dense_features] = all_dense_concat[len(df):len(all_dense_concat),0:len(dense_features)]

        # 构造模型所需要的参数
#         args=get_args()
#         cur_sparse_features=sparse_features+['user_{}_sum_group'.format(action),'user_{}_mean_group'.format(action)]
        cur_sparse_features=sparse_features
        sparse_feature_dict=dict()
        for feat in cur_sparse_features:
            sparse_feature_dict[feat]=df[feat].max()+1
        dense_feature_list=dense_features
    
        text_feature_dict=dict()
#         cur_text_features=text_features+['user_{}_history'.format(action)]
        cur_text_features=text_features
        for k in cur_text_features:
            text_feature_dict[k]=all_text_feature_dict[k]
    
    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        pretrain_data=load_pretrain(args.load_flag,ROOT_PATH+'/pretrain/pretrain_'+action+'.npz')
        train,val=make_train_val(df,day=15)
        train=train.sample(frac=1.0)
        print(train.shape)
        userid_list = val['userid'].astype(str).tolist() # val中所有uid列表 计算auc需要使用
        val_labels=val[action].values

        batch_list=convert_feed_list(train,cur_sparse_features+dense_features+cur_text_features,args.batch_size,action)
#         pretrain_data=load_pretrain(args.load_flag,ROOT_PATH+'/pretrain/pretrain_'+action+'.npz')
        # 2 循环repeat次模型 得到sub文件
        for i in range(args.repeat):
            # 选择lr
#             args.lr=lr_list[args.repeat%len(lr_list)]
            tf.set_random_seed(seed_list[args.repeat%len(seed_list)])
            # 2-1 构造模型
            model=MyModel(args,sparse_feature_dict,dense_feature_list,text_feature_dict, pretrain_data)
        
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())

            # 2-2 sample 此次的epochs数
            epochs=random.choice(action_epochs[action])
            print('********repeat:{},epochs:{}*****************'.format(i+1,epochs))
            t0=time.time()
            # 2-3 训练epochs次
            #batch_list=convert_feed_list(train,cur_sparse_features+dense_features+cur_text_features,args.batch_size,action)
            for epoch in range(epochs):
                loss,task_loss=0.,0.
#                 train=train.sample(frac=1.0)
                n_batch=train.shape[0]//args.batch_size+1
                
                for idx in range(n_batch):
                    values_dict=batch_list[idx]
#                     batch_data=get_batch_data(train,args.batch_size,idx)
#                     values_dict=convert_feed_dict(batch_data,cur_sparse_features+dense_features+cur_text_features,action)
                    feed_dict=model._get_feed_dict(values_dict)
                    _,batch_loss,batch_task_loss,batch_reg_loss=model.train(sess,feed_dict)
                    loss+=batch_loss
                    task_loss+=batch_task_loss
        
                loss/=n_batch
                task_loss/=n_batch
                #print('epoch:{},loss:{:.5f},task_loss:{:.5f}'.format(epoch+1,loss,task_loss))
                t1=time.time()
                print('epoch:{},loss:{:.5f},task_loss:{:.5f},cost time:{:.2f}s'.format(epoch+1,loss,task_loss,t1-t0))
                # 2-4 训练完之后 保存预测结果
                if(epoch==epochs-1):
                    cur=predict_by_batch(model,sess,test,args.batch_size*4,
                                             cur_sparse_features+dense_features+cur_text_features,action)
                    if(i==0):
                        predict_best[action]=cur
                    else:
                        predict_best[action]=predict_best[action]+cur

    # 3 predict文件取平均 保存
    for action,predict in predict_best.items():
        sub_predict[action]=predict/args.repeat
    sub_predict.to_csv('{}/b_sub_{}_{}.csv'.format(SUB_PATH,args.model_name,args.repeat),index=False)
    print('save ok!')


if __name__=='__main__':
    main()
            
