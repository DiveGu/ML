import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


class MyModel():

    # args是参数字典 
    def __init__(self,args,sparse_feature_dict,dense_feature_list,
                 text_feature_dict,pretrain_data):
        
        self.lr=args.lr
        self.emb_dim=args.emb_dim
        self.dnn_layer=eval(args.dnn_layer)
        self.word_size=args.word_size
        self.word_dim=args.word_dim
        self.tag_size=args.tag_size
        self.keyword_size=args.keyword_size
        self.pretrain_data=pretrain_data
        self.load_flag=args.load_flag
        
        self.reg_1=args.reg_1 # emb的正则化系数

        self.filter_num=args.filter_num # 每次卷积的核数量
        self.filter_size=eval(args.filter_size) # 卷积的步长 list
        
        self.sparse_feature_dict=sparse_feature_dict
        self.dense_feature_list=dense_feature_list
        self.text_feature_dict=text_feature_dict
        
        # 1 初始化所有模型参数
        self.weights_dict=self._init_weights()
        # 1 定义输入placeholder
        self.input_dict=self._init_input()
        # 2 搭建模型
        self.predict=self._forward()
        # 3 计算损失函数
        self.task_loss,self.reg_loss=self._get_loss(self.predict,self.input_dict['target'])
        self.loss=self.task_loss+self.reg_loss
        # 4 优化器
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    # 初始化嵌入表和w
    def _init_weights(self):
        weights_dict=dict()
        self.initializer=tensorflow.contrib.layers.xavier_initializer()
        # 各个sparse feature的嵌入表 name:max_size
        for feature_name,size in self.sparse_feature_dict.items():
            if(feature_name=='feedid' and self.load_flag):
                load_path='/testcbd017_gujinfang/GJFCode/WeChat_2021/Code/data/wechat_algo_data1/feed_emb_give.npz'
                feed_emb_give=tf.Variable(initial_value=np.load(load_path)['feed_emb_give'], trainable=False,
                                                    name='feed_emb_id_give_embedding', dtype=tf.float32)
                weights_dict[feature_name]=tf.layers.dense(feed_emb_give,self.emb_dim,use_bias=False) #[N,16]
            else:
                weights_dict[feature_name]=tf.Variable(self.initializer([size,self.emb_dim]),
                                                      name='{}_embedding'.format(feature_name))
        # 物品预训练嵌入
        if(self.pretrain_data is not None):
            weights_dict['feedid']=tf.Variable(initial_value=np.squeeze(self.pretrain_data['feedid_emb'],axis=0), trainable=False,
                                                    name='feedid_embedding', dtype=tf.float32)
            
        # 词嵌入
        weights_dict['word']=tf.Variable(self.initializer([self.word_size,self.word_dim]),
                                                  name='word_embedding')
        
        # 卷积核
        for size in self.filter_size:
            for txt_name in self.text_feature_dict.keys():
                if(txt_name in ['des_words','ocr_words','asr_words']):
                    weights_dict['{}_filter_w_{}'.format(txt_name,size)]=tf.Variable(
                        tf.truncated_normal([size,self.word_dim,1,self.filter_num],stddev=0.1),
                                                                         name = "{}_filte_weights_{}".format(txt_name,size))
                    weights_dict['{}_filter_b_{}'.format(txt_name,size)]=tf.Variable(tf.constant(0.1, shape=[self.filter_num]),
                                                                         name="{}_filter_bias_{}".format(txt_name,size))
                if(txt_name in ['user_watch_history']):
                    weights_dict['{}_filter_w_{}'.format(txt_name,size)]=tf.Variable(
                        tf.truncated_normal([size,self.emb_dim,1,self.filter_num],stddev=0.1),
                                                                         name = "{}_filte_weights_{}".format(txt_name,size))
                    weights_dict['{}_filter_b_{}'.format(txt_name,size)]=tf.Variable(tf.constant(0.1, shape=[self.filter_num]),
                                                                         name="{}_filter_bias_{}".format(txt_name,size))
                
                    
        # 标签 关键字的嵌入
        weights_dict['tag']=tf.Variable(self.initializer([self.tag_size,self.emb_dim]),
                                                  name='tag_embedding')
        weights_dict['keyword']=tf.Variable(self.initializer([self.keyword_size,self.emb_dim]),
                                                  name='keyword_embedding')
                    

        # 线性特征参数
        if(len(self.dense_feature_list)>0):
            weights_dict['liner']=tf.Variable(self.initializer([len(self.dense_feature_list),1]),
                                                      name='dense_w')
        
        # 卷积之后feature map加mlp成feature
        weights_dict['txt_feature_w']=tf.Variable(self.initializer([self.filter_num*len(self.filter_size),self.emb_dim]),
                                                  name='txt_feature_w')
        weights_dict['txt_feature_b']=tf.Variable(self.initializer([self.emb_dim]),
                                                  name='txt_feature_b')
        


        return weights_dict

    # 定义模型输入placeholder
    def _init_input(self):
        input_dict=dict()
        # 分类特征 id类
        for feature_name in self.sparse_feature_dict.keys():
            input_dict[feature_name]=tf.placeholder(tf.int32,shape=(None,),name=feature_name)

        # 数值特征
        for feature_name in self.dense_feature_list:
            input_dict[feature_name]=tf.placeholder(tf.float32,shape=(None,),name=feature_name)

        # 文本特征
        for feature_name,size in self.text_feature_dict.items():
            input_dict[feature_name]=tf.placeholder(tf.int32,shape=[None,size],name=feature_name)

        input_dict['dnn_keep_prob']=tf.placeholder(tf.float32,name='dnn_keep_prob')
        
        # y
        input_dict['target']=tf.placeholder(tf.int32,shape=(None,),name='target')
        return input_dict
    
    # 将input_dict中的所有变量做key 传真实values
    def _get_feed_dict(self,values_dict):
        return dict(zip(self.input_dict.values(),values_dict.values()))

    # tag keyword令pad嵌入一直为0
    def _get_pad_embedding(self,emb,k,ids):
        pad_emb=tf.Variable(tf.zeros([1,k]))
#         pad_emb=tf.reduce_mean(emb,axis=0,keepdims=True) # [1,k]
        pad_emb=tf.concat([pad_emb,emb],axis=0)
        return tf.nn.embedding_lookup(pad_emb,ids)

    # userid 和target item的各种内容进行交互
    def _uid_pair_list(self,uid,pair_list):
        pair_ret=[]
        for id in pair_list:
            tmp=tf.multiply(uid,id)
            pair_ret.append(tmp)
        pair_ret=tf.concat(pair_ret,axis=1)
        return pair_ret
    
    # 构造模型
    def _forward(self):
        # 1 所有id查嵌入进行拼接
        self.embeddings_list=[]
        for id in self.sparse_feature_dict.keys():
            tmp=tf.nn.embedding_lookup(self.weights_dict[id],self.input_dict[id]) # [N,k]
            self.embeddings_list.append(tmp)
            
        self.embeddings=tf.concat(self.embeddings_list,axis=1) # [N,mK]
        
        self.pair_interaction=[]
        # 标签pooling
        if('manual_tag' in self.text_feature_dict.keys()):
            self.tag1_embeddings=self._get_pad_embedding(self.weights_dict['tag'],self.emb_dim,self.input_dict['manual_tag'])
            self.tag2_embeddings=self._get_pad_embedding(self.weights_dict['tag'],self.emb_dim,self.input_dict['machine_tag'])
            self.tag_embeddings=tf.concat([self.tag1_embeddings,self.tag2_embeddings],axis=1) # [N,10,k]
            self.tag_pooling=tf.reduce_mean(self.tag_embeddings,axis=1,keepdims=False) # [N,k]

            
            self.embeddings=tf.concat([self.embeddings,self.tag_pooling],axis=1)
            self.pair_interaction.append(tf.multiply(tf.reduce_mean(self.tag1_embeddings,axis=1,keepdims=False),
                                                     tf.reduce_mean(self.tag2_embeddings,axis=1,keepdims=False)))

        
        
        if('manual_keywords' in self.text_feature_dict.keys()):
            #self.keyword1_embeddings=tf.nn.embedding_lookup(self.weights_dict['keyword'],self.input_dict['manual_keywords']) # [N,4,k]
            #self.keyword2_embeddings=tf.nn.embedding_lookup(self.weights_dict['keyword'],self.input_dict['machine_keywords']) # [N,6,k]
            
            self.keyword1_embeddings=self._get_pad_embedding(self.weights_dict['keyword'],self.emb_dim,self.input_dict['manual_keywords'])
            self.keyword2_embeddings=self._get_pad_embedding(self.weights_dict['keyword'],self.emb_dim,self.input_dict['machine_keywords'])
            self.keyword_embeddings=tf.concat([self.keyword1_embeddings,self.keyword2_embeddings],axis=1) # [N,10,k]
            self.keyword_pooling=tf.reduce_mean(self.keyword_embeddings,axis=1,keepdims=False) # [N,k]
            
            self.embeddings=tf.concat([self.embeddings,self.keyword_pooling],axis=1)
            self.pair_interaction.append(tf.multiply(tf.reduce_mean(self.keyword1_embeddings,axis=1,keepdims=False),
                                                     tf.reduce_mean(self.keyword2_embeddings,axis=1,keepdims=False)))
        
        

        # TODO 文本特征提取          
        self.txt_feature=[]
        # 先单个des_text 文本特征提取
        for feature_name,size in self.text_feature_dict.items():
            if(feature_name=='des_words'):
                self.des_embedding=tf.nn.embedding_lookup(self.weights_dict['word'],self.input_dict['des_words']) # [N,50,k]
                self.des_embedding=tf.expand_dims(self.des_embedding,-1) # [N,50,k,1]
                self.des_feature=self._text_cnn(self.des_embedding,'des_words') # [N,32]
                self.des_feature=tf.layers.dense(self.des_feature,self.emb_dim,activation='relu') # [N,16]
#                 self.des_feature=tf.matmul(self.des_feature,self.weights_dict['txt_feature_w'])
#                 self.des_feature=tf.nn.relu(self.des_feature+self.weights_dict['txt_feature_b'])
    
                self.txt_feature.append(self.des_feature)
            elif(feature_name=='ocr_words'):
                self.ocr_embedding=tf.nn.embedding_lookup(self.weights_dict['word'],self.input_dict['ocr_words']) # [N,100,k]
                self.ocr_embedding=tf.expand_dims(self.ocr_embedding,-1) # [N,100,k,1]
                self.ocr_feature=self._text_cnn(self.ocr_embedding,'ocr_words') # [N,32]
#                 self.ocr_feature=tf.layers.dense(self.ocr_feature,self.emb_dim,activation='relu') # [N,16]
                self.ocr_feature=tf.matmul(self.ocr_feature,self.weights_dict['txt_feature_w'])
                self.ocr_feature=tf.nn.relu(self.ocr_feature+self.weights_dict['txt_feature_b'])
                self.txt_feature.append(self.ocr_feature)
            elif(feature_name=='asr_words'):
                self.asr_embedding=tf.nn.embedding_lookup(self.weights_dict['word'],self.input_dict['asr_words']) # [N,100,k]
                self.asr_embedding=tf.expand_dims(self.asr_embedding,-1) # [N,100,k,1]
                self.asr_feature=self._text_cnn(self.asr_embedding,'asr_words') # [N,32]
#                 self.asr_feature=tf.layers.dense(self.asr_feature,self.emb_dim,activation='relu') # [N,16]
                self.asr_feature=tf.matmul(self.asr_feature,self.weights_dict['txt_feature_w'])
                self.asr_feature=tf.nn.relu(self.asr_feature+self.weights_dict['txt_feature_b'])
                self.txt_feature.append(self.asr_feature)
            elif(feature_name=='user_watch_history'):
                self.watch_embedding=tf.nn.embedding_lookup(self.weights_dict['feedid'],self.input_dict['user_watch_history']) # [N,100,k]
#                 self.watch_feature=tf.reduce_mean(self.watch_embedding,axis=1)
# #                 self.watch_embedding=tf.expand_dims(self.watch_embedding,-1) # [N,100,k,1]
# #                 self.watch_feature=self._text_cnn(self.watch_embedding,'user_watch_history') # [N,32]
# #                 self.watch_feature=tf.layers.dense(self.watch_feature,self.emb_dim,activation='relu') # [N,16]
#                 self.txt_feature.append(self.watch_feature)


#         self.txt_pair=self._pair_fm(self.txt_feature,3)
        if('des_words' in self.text_feature_dict.keys()):
            self.txt_feature=tf.concat(self.txt_feature,axis=1) # [N,48]
        

        # 2 liner侧
        if(len(self.dense_feature_list)>0):
            self.liner_input=[tf.expand_dims(self.input_dict[tmp],axis=1) for tmp in self.dense_feature_list] # [N,1]
            self.liner_input=tf.concat(self.liner_input,axis=1)# [N,n]
            self.liner_ouput=tf.layers.dense(self.liner_input,1) # [N,1]
        
        # 3 传入NN

        # 3-1 : 构造dnn input,id类+liner
        self.concat_input=self.embeddings
        if(len(self.dense_feature_list)>0):
            self.concat_input=tf.concat([self.concat_input,self.liner_input],axis=1)
        # 3-2 : 构造dnn input, txt feature
        use_text=('des_words' in self.text_feature_dict.keys())
        if(use_text):
            self.concat_input=tf.concat([self.concat_input,self.txt_feature],axis=1)
#             self.concat_input=tf.concat([self.concat_input,self.txt_pair],axis=1)
            
        # 3-3 : 构造dnn input，是否对于id类显式两两建模
        pair_flag=True
        if(pair_flag):
            field_num=len(self.sparse_feature_dict)
            if('manual_tag' in self.text_feature_dict.keys()):
                field_num+=1
            if('manual_keywords' in self.text_feature_dict.keys()):
                field_num+=1
                
            self.concat_input_emb=self.embeddings
            self.fields=tf.reshape(self.concat_input_emb,
                                   shape=[-1,field_num,self.emb_dim]) # [N,7,16]
            
            # 写法一：直接fm+dnnoutput上

            # 写法二：将fm弄成[N,k]，concat，送入dnn
            self.sum_square=tf.reduce_sum(self.fields, axis=1,keepdims=False) #[N,16]
            self.sum_square=tf.square(self.sum_square) # [N,16]
            
            self.square_sum=tf.square(self.fields) # [N,7,16]
            self.square_sum=tf.reduce_sum(self.square_sum,axis=1,keepdims=False) # [N,16]
            
            self.second_order=0.5*tf.subtract(self.sum_square,self.square_sum) # [N,16]
            
            self.concat_input=tf.concat([self.concat_input,self.second_order],axis=1)

#             # uid和所有特征类 进行交互
#             self.uid_pair=self._uid_pair_list(self.embeddings_list[0],self.embeddings_list[1:]+[self.des_feature]+[self.tag_pooling,self.keyword_pooling])
# #             # feedid和所有特征类 进行交互
# # #             self.feedid_pair=self._uid_pair_list(self.embeddings_list[1],self.embeddings_list[2:]+[self.des_feature]+[self.tag_pooling,self.keyword_pooling])
#             self.concat_input=tf.concat([self.concat_input,self.uid_pair],axis=1)

            if(len(self.pair_interaction)>0):
                # tag keyword des 三者交互
                self.txt_key_tag=tf.multiply(self.keyword_pooling,self.tag_pooling)
                self.txt_key_tag=tf.multiply(self.txt_key_tag,self.des_feature)
                
                # tag keyword两两独立手工和机器显式交叉
                self.pair_interaction=tf.concat(self.pair_interaction,axis=1) # [N,?k]
                self.concat_input=tf.concat([self.concat_input,self.pair_interaction,self.txt_key_tag],axis=1)
                

        
        # HISTORY LIST pool
        if('user_watch_history' in self.text_feature_dict.keys()):
            self.history_pool=self._history_pool()
#             self.concat_input=tf.concat([self.concat_input,self.history_pool],axis=1)
            self.concat_input=tf.concat([self.concat_input,self.history_pool,tf.multiply(self.history_pool,self.embeddings_list[1])],axis=1)
#             self.concat_input=tf.concat([self.concat_input,tf.multiply(self.history_pool,self.embeddings_list[0])],axis=1)
                

        # 3-4 多层dnn        
        self.concat_output=self.concat_input
            
        for layer_size in self.dnn_layer:
            self.concat_output=tf.layers.dense(self.concat_output,layer_size,activation='relu')
#             self.concat_output=tf.layers.dense(self.concat_output,layer_size,
#                                                activation='relu',kernel_initializer=self.initializer)
        
        # 最后一个dnn 加上dropout
        self.concat_output=tf.nn.dropout(self.concat_output, keep_prob=self.input_dict['dnn_keep_prob'])
            
        self.dnn_output=tf.layers.dense(self.concat_output,1) # [N,1]

#         # 4 预测最后一层
#         if(len(self.dense_feature_list)>0):
#             self.logit=tf.add(self.dnn_output,self.liner_ouput) # [N,1]
#         else:
#             self.logit=self.dnn_output
        self.logit=self.dnn_output
            
        self.logit=tf.squeeze(self.logit,axis=1) # (N,)
        self.predict=tf.nn.sigmoid(self.logit)
        return self.predict


    # 历史id的集合 pool
    def _history_pool(self):
        diff_action_ist=[]
        feed_target=tf.nn.embedding_lookup(self.weights_dict['feedid'],self.input_dict['feedid']) # [N,k]
#         ["read_comment", "like", "click_avatar",  "forward"]
        if('user_read_comment_history' in self.text_feature_dict.keys()):
            tmp=tf.nn.embedding_lookup(self.weights_dict['feedid'],self.input_dict['user_read_comment_history']) # [N,10,k]
#             tmp=tf.reduce_mean(tmp,axis=1,keepdims=False)
            tmp=self._history_gru(tmp)
            diff_action_ist.append(tmp)
#             diff_action_ist.append(tf.multiply(feed_target,tmp))
           
        if('user_like_history' in self.text_feature_dict.keys()):
            tmp=tf.nn.embedding_lookup(self.weights_dict['feedid'],self.input_dict['user_like_history']) # [N,10,k]
#             tmp=tf.reduce_mean(tmp,axis=1,keepdims=False)
            tmp=self._history_gru(tmp)
            diff_action_ist.append(tmp)
#             diff_action_ist.append(tf.multiply(feed_target,tmp))
           
        if('user_click_avatar_history' in self.text_feature_dict.keys()):
            tmp=tf.nn.embedding_lookup(self.weights_dict['feedid'],self.input_dict['user_click_avatar_history']) # [N,10,k]
#             tmp=tf.reduce_mean(tmp,axis=1,keepdims=False)
            tmp=self._history_gru(tmp)
            diff_action_ist.append(tmp)
#             diff_action_ist.append(tf.multiply(feed_target,tmp))
            
        if('user_forward_history' in self.text_feature_dict.keys()):
            tmp=tf.nn.embedding_lookup(self.weights_dict['feedid'],self.input_dict['user_forward_history']) # [N,10,k]
#             tmp=tf.reduce_mean(tmp,axis=1,keepdims=False)
            tmp=self._history_gru(tmp)
            diff_action_ist.append(tmp)
#             diff_action_ist.append(tf.multiply(feed_target,tmp))

        if('user_watch_history' in self.text_feature_dict.keys()):
            tmp=tf.nn.embedding_lookup(self.weights_dict['feedid'],self.input_dict['user_watch_history']) # [N,10,k]
            # self.embeddings_list[0]
#             user_query=tf.layers.dense(self.embeddings_list[0],self.emb_dim,activation='tanh')
#             tmp=self._target_attention(tmp,user_query)
#             feed_query=tf.layers.dense(feed_target,self.emb_dim,activation='tanh')
#             tmp=self._target_attention(tmp,feed_query)
            
#             tmp=tf.reduce_mean(tmp,axis=1,keepdims=False)
#             tmp=self._history_gru(tmp)

            tmp=tf.expand_dims(tmp,-1) # [N,20,k,1]
            tmp=self._text_cnn(tmp,'user_watch_history') # [N,32]
            tmp=tf.layers.dense(tmp,self.emb_dim,activation='relu') # [N,16]
            diff_action_ist.append(tmp)
#             diff_action_ist.append(tf.multiply(feed_target,tmp))
            
        return tf.concat(diff_action_ist,axis=1)
    
    # gru 
    def _history_gru(self,inputs):
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=16)
#         init_state = gru_cell.zero_state(512, dtype=tf.float32)
#         output, state = tf.nn.dynamic_rnn(gru_cell, inputs, initial_state=init_state, time_major=False)
        output, state = tf.nn.dynamic_rnn(gru_cell, inputs, dtype=tf.float32, time_major=False)    
        return state
    
    # din注意力聚合
    def _target_attention(self,his,target):
        # his [N,20,k] 
        # target [N,k]
        # 1 计算score
        target=tf.expand_dims(target,axis=1) # [N,1,k] 
        score=tf.multiply(target,his) # [N,20,k]
        score=tf.reduce_sum(score,axis=2,keep_dims=False) # [N,20]
        score=tf.nn.softmax(score,axis=1) # [N,20]
        score=tf.expand_dims(score,axis=-1) # [N,20,1]
        # 2 score聚合his
        combine=tf.multiply(score,his) # [N,20,k]
        combine=tf.reduce_sum(combine,axis=1,keep_dims=False) # [N,k]
        return combine
        
    
    # pair构造group的fm交互
    def _pair_fm(self,group,num):
        fields=tf.reshape(group,
                          shape=[-1,num,self.emb_dim]) # [N,3,16]
        sum_square=tf.reduce_sum(fields, axis=1,keepdims=False) #[N,16]
        sum_square=tf.square(sum_square) # [N,16]
            
        square_sum=tf.square(fields) # [N,3,16]
        square_sum=tf.reduce_sum(square_sum,axis=1,keepdims=False) # [N,16]
            
        second_order=0.5*tf.subtract(sum_square,square_sum) # [N,16]
            
        return second_order
    
    
    # 在input上进行text cnn
    def _text_cnn(self,input,feat_name):
        # [N,50,k,1]
        feat_list=[]
        for size in self.filter_size:
            # 1 卷积层 -> [N,49,16,8]
            conv1 = tf.nn.conv2d(input=input,
                                 filter=self.weights_dict['{}_filter_w_{}'.format(feat_name,size)], # 卷积核[size,k,1,num]
                                 strides=[1,1,1,1],# 不用管这个维度
                                 padding='VALID',) 
        
            # 2 激活函数->[N,49,1,8]
            conv1=tf.nn.relu(tf.nn.bias_add(conv1,self.weights_dict['{}_filter_b_{}'.format(feat_name,size)])) 
#             print(conv1.get_shape())
            # 3 最大池化 -> [N,1,1,8]
            pool1 = tf.nn.max_pool(value=conv1, 
                                   ksize=[1, self.text_feature_dict[feat_name]-size+1, 1, 1],  
                                   strides=[1, 1, 1, 1], # 不用管这个维度
                                   padding='VALID',) # [N,1,1,num]
            
#             print(pool1.get_shape())
            feat_list.append(pool1)
    
            
        des_feature=tf.concat(feat_list,axis=3) # [N,1,1,24]
        max_num=len(self.filter_size)*self.filter_num
        des_feature=tf.squeeze(des_feature,axis=1)
        des_feature=tf.squeeze(des_feature,axis=1)
#         print(des_feature.get_shape())
        
        return des_feature
            


    # 计算损失函数
    def _get_loss(self,predict,target):
        task_loss = tf.losses.log_loss(target, predict)
        
        reg_loss=tf.nn.l2_loss(self.embeddings)
        if('des_words' in self.text_feature_dict.keys()):
            reg_loss+=tf.nn.l2_loss(self.des_embedding)
        if('ocr_words' in self.text_feature_dict.keys()):
            reg_loss+=tf.nn.l2_loss(self.ocr_embedding)
        if('asr_words' in self.text_feature_dict.keys()):
            reg_loss+=tf.nn.l2_loss(self.asr_embedding)
        if('manual_tag' in self.text_feature_dict.keys()):
            reg_loss+=tf.nn.l2_loss(self.tag1_embeddings)
            reg_loss+=tf.nn.l2_loss(self.tag2_embeddings)
        if('manual_keywords' in self.text_feature_dict.keys()):
            reg_loss+=tf.nn.l2_loss(self.keyword1_embeddings)
            reg_loss+=tf.nn.l2_loss(self.keyword2_embeddings)
            
        #reg_loss=tf.nn.l2_loss(self.concat_input)+tf.nn.l2_loss(self.des_embedding)+tf.nn.l2_loss(self.asr_embedding)
        reg_loss=reg_loss*self.reg_1
        return task_loss,reg_loss

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss,self.task_loss,self.reg_loss],feed_dict)

    # 得到预测
    def get_predict(self,sess,feed_dict):
        tmp=sess.run(self.predict,feed_dict)
        return tmp
    
    # 保存 item 嵌入
    def save_emb(self,sess,save_path):
        feedid_emb=sess.run([self.weights_dict['feedid']],feed_dict={})
        np.savez(save_path,feedid_emb=feedid_emb)
        print('item emb save ok!')
        
