import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


class MyModel():

    # args是参数字典 
    def __init__(self,args,sparse_feature_dict,dense_feature_list,
                 text_feature_dict):
        
        self.lr=args.lr
        self.emb_dim=args.emb_dim
        self.dnn_layer=eval(args.dnn_layer)
        self.word_size=args.word_size
        self.word_dim=args.word_dim
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
            weights_dict[feature_name]=tf.Variable(self.initializer([size,self.emb_dim]),
                                                  name='{}_embedding'.format(feature_name))
        # 词嵌入
        weights_dict['word']=tf.Variable(self.initializer([self.word_size,self.word_dim]),
                                                  name='word_embedding')
        
        # 卷积核
        for size in self.filter_size:
            for txt_name in self.text_feature_dict.keys():
                weights_dict['{}_filter_w_{}'.format(txt_name,size)]=tf.Variable(tf.truncated_normal([size,self.word_dim,1,self.filter_num],stddev=0.1),
                                                                     name = "{}_filte_weights_{}".format(txt_name,size))
                weights_dict['{}_filter_b_{}'.format(txt_name,size)]=tf.Variable(tf.constant(0.1, shape=[self.filter_num]),
                                                                     name="{}_filter_bias_{}".format(txt_name,size))

        # 线性特征参数
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

    # 构造模型
    def _forward(self):
        # 1 所有id查嵌入进行拼接
        self.embeddings=[]
        for id in self.sparse_feature_dict.keys():
            tmp=tf.nn.embedding_lookup(self.weights_dict[id],self.input_dict[id]) # [N,k]
            self.embeddings.append(tmp)

        self.embeddings=tf.concat(self.embeddings,axis=1) # [N,mK]
        

        # TODO 文本特征提取          
        self.txt_feature=[]
        # 先单个des_text 文本特征提取
        for feature_name,size in self.text_feature_dict.items():
            if(feature_name=='des_words'):
                self.des_embedding=tf.nn.embedding_lookup(self.weights_dict['word'],self.input_dict['des_words']) # [N,50,k]
                self.des_embedding=tf.expand_dims(self.des_embedding,-1) # [N,50,k,1]
                self.des_feature=self._text_cnn(self.des_embedding,'des_words') # [N,32]
#                 self.des_feature=tf.layers.dense(self.des_feature,self.emb_dim,activation='relu') # [N,16]
                self.des_feature=tf.matmul(self.des_feature,self.weights_dict['txt_feature_w'])
                self.des_feature=tf.nn.relu(self.des_feature+self.weights_dict['txt_feature_b'])
    
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


#         self.txt_pair=self._pair_fm(self.txt_feature,3)
        #self.txt_feature=tf.concat(self.txt_feature,axis=1) # [N,48]
        
        tmp=self.txt_feature
        for i in range(len(tmp)):
            if(i==0):
                self.txt_feature=tmp[0]
            else:
                self.txt_feature=tf.add(self.txt_feature,tmp[i])
                
        #self.txt_feature=tf.reduce_sum(self.txt_feature,axis=1) # [N,16]

        # 2 liner侧
        self.liner_input=[self.input_dict[tmp] for tmp in self.dense_feature_list]
        self.liner_input=tf.expand_dims(tf.concat(self.liner_input,axis=1),axis=1) # [N,n]
        self.liner_ouput=tf.layers.dense(self.liner_input,1) # [N,1]
        
        # 3 传入NN

        # 3-1 : 构造dnn input,id类+liner
        self.concat_input=tf.concat([self.embeddings,self.liner_input],axis=1)
        # 3-2 : 构造dnn input, txt feature
        use_text=True
        if(use_text):
            self.concat_input=tf.concat([self.concat_input,self.txt_feature],axis=1)
#             self.concat_input=tf.concat([self.concat_input,self.txt_pair],axis=1)
            
        # 3-3 : 构造dnn input，是否对于id类显式两两建模
        pair_flag=True
        if(pair_flag):
            self.concat_input_emb=self.embeddings
            self.fields=tf.reshape(self.concat_input_emb,
                                   shape=[-1,len(self.sparse_feature_dict),self.emb_dim]) # [N,7,16]
            
# 写法一：直接fm+dnnoutput上
#             self.sum_square=tf.reduce_sum(self.fields, axis=1,keepdims=False) #[N,16]
#             self.sum_square=tf.reduce_sum(tf.square(self.sum_square),axis=1,keepdims=True) # [N,1]
            
#             self.square_sum=tf.square(self.fields) # [N,7,16]
#             self.square_sum=tf.reduce_sum(self.square_sum,axis=2,keepdims=False) # [N,7]
#             self.square_sum=tf.reduce_sum(self.square_sum,axis=1,keepdims=True) # [N,1]
            
#             self.second_order=0.5*tf.subtract(self.sum_square,self.square_sum) # [N,1]

            # 写法二：将fm弄成[N,k]，concat，送入dnn
            self.sum_square=tf.reduce_sum(self.fields, axis=1,keepdims=False) #[N,16]
            self.sum_square=tf.square(self.sum_square) # [N,16]
            
            self.square_sum=tf.square(self.fields) # [N,7,16]
            self.square_sum=tf.reduce_sum(self.square_sum,axis=1,keepdims=False) # [N,16]
            
            self.second_order=0.5*tf.subtract(self.sum_square,self.square_sum) # [N,16]
            
            self.concat_input=tf.concat([self.concat_input,self.second_order],axis=1)

        # 3-4 多层dnn        
        self.concat_output=self.concat_input
            
        for layer_size in self.dnn_layer:
            self.concat_output=tf.layers.dense(self.concat_output,layer_size,activation='relu')
#             self.concat_output=tf.layers.dense(self.concat_output,layer_size,
#                                                activation='relu',kernel_initializer=self.initializer)
        
        # 最后一个dnn 加上dropout
        self.concat_output=tf.nn.dropout(self.concat_output, keep_prob=self.input_dict['dnn_keep_prob'])
            
        self.dnn_output=tf.layers.dense(self.concat_output,1) # [N,1]

        # 4 预测最后一层
        self.logit=tf.add(self.dnn_output,self.liner_ouput) # [N,1]
#         if(pair_flag):
#             self.logit=tf.add(self.logit,self.second_order) # [N,1]
            
        self.logit=tf.squeeze(self.logit,axis=1) # (N,)
        self.predict=tf.nn.sigmoid(self.logit)
        return self.predict


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

