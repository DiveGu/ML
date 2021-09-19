import os
import pandas as pd
import numpy as np
import tensorflow as tf

from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from mmoe import MMOE
from evaluation_v2 import evaluate_deepctr

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ACTION_PATH='/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data/wechat_algo_data1/user_action.csv'
FEED_PATH='/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data/wechat_algo_data1/feed_info.csv'
TEST_PATH='/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data/wechat_algo_data1/test_a.csv'
SUB_PATH='/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data/result.csv'

if __name__ == "__main__":
    epochs = 20
    batch_size = 512
    embedding_dim = 16
    target = ["read_comment", "like", "click_avatar", "forward"]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    #sparse_features = ['userid', 'feedid','bgm_song_id']
    dense_features = ['videoplayseconds', ]

    data = pd.read_csv(ACTION_PATH)

    feed = pd.read_csv(FEED_PATH)
    feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
    data = data.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')

    test = pd.read_csv(TEST_PATH)
    test = test.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                      on='feedid')

    # 1.fill nan dense_feature and do simple Transformation for dense features
    data[dense_features] = data[dense_features].fillna(0, )
    test[dense_features] = test[dense_features].fillna(0, )

    data[dense_features] = np.log(data[dense_features] + 1.0)
    test[dense_features] = np.log(test[dense_features] + 1.0)

    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())

    train = data[(data['date_'] < 14)&(data['date_'] >= 7) ]
    val = data[data['date_'] == 14]  # 第14天样本作为验证集

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    # 4.Define Model,train,predict and evaluate
    best_weight_auc=0.0
    train_model = MMOE(dnn_feature_columns, num_tasks=4, expert_dim=8, num_experts=8,
                       dnn_hidden_units=(128, 128,64), seed=1997, 
                       tasks=['binary', 'binary', 'binary', 'binary'])
    train_model.compile("adam", loss='binary_crossentropy')
    # print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=0)

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        weight_auc=evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)
        print(f'epoch{epoch+1},weight_auc{weight_auc}')
        if(weight_auc>best_weight_auc):
          pred_ans= train_model.predict(test_model_input, batch_size=batch_size *20)
          # 5.生成提交文件
          for i, action in enumerate(target):
              test[action] = pred_ans[i]
          sub_path='/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data/submit/result_mmoe_{}.csv'.format(weight_auc)
          test[['userid', 'feedid'] + target].to_csv(sub_path, index=None, float_format='%.6f')
          print(f'{epoch+1}to_csv ok')
          best_weight_auc=weight_auc
          

    t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    t2 = time()
    print('4 behavior %d cost time(ms):%.3f' % (len(test), (t2 - t1) * 1000.0))
    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    print('4 behavior 2000 cost time(ms):%.3f' % ts)

    # 5.生成提交文件
  #  for i, action in enumerate(target):
  #      test[action] = pred_ans[i]
  #  test[['userid', 'feedid'] + target].to_csv(SUB_PATH, index=None, float_format='%.6f')
  #  print('to_csv ok')
