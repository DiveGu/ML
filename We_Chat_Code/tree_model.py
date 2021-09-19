import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc
import time

pd.set_option('display.max_columns', None)

# 原始数据路径
ROOT_PATH = "/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data"
DATA_PATH=ROOT_PATH+'/wechat_algo_data1'
SAVE_PATH=DATA_PATH+'/tree_feature'
SUB_PATH=ROOT_PATH+'/submit'

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
# y_list = ['read_comment', 'like', 'click_avatar', 'forward']
max_day = 15

## 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])

    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)

    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break

        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0

    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0

    user_auc = float(total_auc)/size
    return user_auc

# 把tag keyword弄成multi-hot
def process_multi_hot(df,col,prefix):
   # print('缺失比例:{}'.format(df[col].isna().mean()))
    print(df.loc[0,col])
    multi_hot_col=df[col].str.replace(';','|').str.get_dummies()
    multi_hot_col.columns=[prefix+'_'+na for na in list(multi_hot_col.columns)]
    return multi_hot_col

def drop_lillte_col(df,rate=0.01):
    drop_cols=[]
    for c in df.columns:
        if(df[c].mean()*100<rate):
            drop_cols.append(c)
    n1,n2=len(df.columns),len(drop_cols)
    print('all cols:{},drop cols:{}'.format(n1,n2))
    return df.drop(columns=drop_cols)

## 基于原始数据进行特征工程、制作出全特征的train test 保存
def data_feature_make():
    ## 读取训练集
    train = pd.read_csv(DATA_PATH+'/user_action.csv')
    
    ## 读取测试集
    test = pd.read_csv(DATA_PATH+'/test_b.csv')
    test['date_'] = max_day

    # 合并处理
    df = pd.concat([train, test], axis=0, ignore_index=True)

    ## 读取视频特征表
    feed_info = pd.read_csv(DATA_PATH+'/feed_feature.csv')

    ## 挑选feed feature列
    feed_info = feed_info[[
        'feedid', 'authorid', 'videoplayseconds','bgm_song_id', 'bgm_singer_id','video_time_group','feed_cluter','manual_tag_list'
    ]]
    
#     multi_tag=process_multi_hot(feed_info,'manual_tag_list','tag')
#     multi_tag=drop_lillte_col(multi_tag,rate=0.05)
#     tag_cols=list(multi_tag.columns)
#     feed_info=feed_info.join(multi_tag)
#     feed_info=feed_info.drop(columns=['manual_tag_list'])

    df = df.merge(feed_info, on='feedid', how='left')
    ## 视频时长是秒，转换成毫秒，才能与play、stay做运算
    df['videoplayseconds'] *= 1000

    ## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
    df['is_finish'] = (df['play'] >= 0.9*df['videoplayseconds']).astype('int8')
    df['play_times'] = df['play'] / df['videoplayseconds']

    play_cols = [
        'is_finish', 'play_times', 'play', 'stay'
    ]

    # 滑窗统计
    ## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
    n_day = 5
    
    stat_cols_list=[['userid'],['feedid'],['authorid'],['bgm_song_id'],['bgm_singer_id'],
                    ['video_time_group'],['feed_cluter'],['userid', 'authorid'],
                    ['userid', 'feed_cluter'],['userid', 'video_time_group']]
#     for tag_c in tag_cols:
#         stat_cols_list.append([tag_c])

    for stat_cols in tqdm(stat_cols_list):

        f = '_'.join(stat_cols)
        stat_df = pd.DataFrame()
        for target_day in range(2, max_day + 1):
            left, right = max(target_day - n_day, 1), target_day - 1
            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
            
            # 使用transform使得输出和输入 数量上对齐 直接用.sum() 其输出相当于set 数量上对不齐
            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
            g = tmp.groupby(stat_cols)
            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

            # 统计近期当前特征列各类 play_times, play, stay
            for x in play_cols[1:]:
                for stat in ['max','mean','median']:
                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
            # 统计各类 近期对于target的比例
            for y in y_list[:2]:
                tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
                tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])

            tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
            del g, tmp

        df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
        del stat_df

        gc.collect()


    ## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行
    all_static_cols=['userid', 'feedid', 'authorid','feed_cluter','video_time_group','bgm_song_id','bgm_singer_id']+tag_cols
    for f in tqdm(all_static_cols):
        df[f + '_count'] = df[f].map(df[df['date_']<14][f].value_counts()) # <14会出现空值
#         df[f + '_count'] = df[f].fillna(0) # <14会出现空值
    

    for f1, f2 in tqdm([
#         ['userid', 'feedid'],
        ['userid', 'authorid']
    ]):

        df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
        df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

    for f1, f2 in tqdm([
        ['userid', 'authorid'],
        ['userid', 'feed_cluter'],
    ]):

        df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
        df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
        df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

    df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
    df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
    df['videoplayseconds_in_feed_cluter'] = df.groupby('feed_cluter')['videoplayseconds'].transform('mean')
    df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')

    ## 内存够用的不需要做这一步
    #df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])

    train = df[~df['read_comment'].isna()].reset_index(drop=True)
    test = df[df['read_comment'].isna()].reset_index(drop=True)
    
#     train.to_csv(DATA_PATH+'/tree_train.csv',index=False)
#     test.to_csv(DATA_PATH+'/tree_test.csv',index=False)
#     print('save ok!')
    return train,test

# 根据path读取全量trian 然后分出真正的train val
def make_train_val(df,day=14):
    # 写法1：前t-1天预测t天
    train=df[(df['date_']<day) & (df['date_']>=day-7)].reset_index(drop=True)
    day=min(day,14)
    val=df[df['date_']==day].reset_index(drop=True)
    return train,val

# 验证集分数
def train_val(train,val,test):
    play_cols = [
        'is_finish', 'play_times', 'play', 'stay'
    ]
    cols = [f for f in train.columns if f not in ['date_'] + play_cols + y_list]
    uauc_list = []
    r_list = []
    
    for y in y_list[:4]:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=200,
            max_depth=8,
            num_leaves=47,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021,
            metric='None'
        )

        clf.fit(
            train[cols], train[y],
            eval_set=[(val[cols], val[y])],
            eval_metric='auc',
            early_stopping_rounds=30,
            verbose=50
        )

        val[y + '_score'] = clf.predict_proba(val[cols])[:, 1]
        val_uauc = uAUC(val[y], val[y + '_score'], val['userid'])
        uauc_list.append(val_uauc)
        print(val_uauc)

        r_list.append(clf.best_iteration_)
        print('runtime: {}\n'.format(time.time() - t))


    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
    print(uauc_list)
    print(weighted_uauc)
    
    ##################### 全量训练 #####################

    train=pd.concat([train, val], axis=0, ignore_index=True)
    r_dict = dict(zip(y_list[:4], r_list))
    for y in y_list[:4]:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=r_dict[y],
            max_depth=8,
            num_leaves=47,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021
        )

        clf.fit(
            train[cols], train[y],
            eval_set=[(train[cols], train[y])],
            early_stopping_rounds=r_dict[y],
            verbose=100
        )

        test[y] = clf.predict_proba(test[cols])[:, 1]
        print('runtime: {}\n'.format(time.time() - t))

    test[['userid', 'feedid'] + y_list[:4]].to_csv(
        SUB_PATH+'/tree_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),
        index=False
    )
    
# 1 制作特征保存全量数据
t0=time.time()
train,test=data_feature_make()
t1=time.time()
print('make concat and save all feature cost time {:.2f}s'.format(t1-t0))

train1,val=make_train_val(train,day=14)
# 3 线下验证
train_val(train1,val,test)
