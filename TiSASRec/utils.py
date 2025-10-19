import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

#随机选择一个负样本
def random_neq_item(left,right,item_had):
    item=np.random.randint(left,right+1)
    while item in item_had:
        item = np.random.randint(left, right+1)
    return item

#计算相对时间 只是单纯的绝对时间戳变成相对时间 没有做其他处理
def cul_re_pos_matrix(time_seq,time_span):
    maxlen = time_seq.shape[0]
    time_matrix = np.zeros([maxlen,maxlen],dtype=np.int32)
    for i in range(maxlen):
        for j in range(maxlen):
            re_time = abs(time_seq[j]-time_seq[i])
            if re_time>time_span:
                time_matrix[i][j]=time_span
            else:
                time_matrix[i][j]=re_time
    return time_matrix

#我还是不懂这里的去掉最后一个是为什么 不去掉行吗
#明明已经划分了train valid test啊
#按道理来说user_train.shape=[user_num,maxlen] 当然也有可能有些user的交互样本过长 所以这不是一个规则的tensor
def relation(user_train,usernum,maxlen,time_span):
    all_relation_matrix=dict()
    for user in tqdm(range(1,usernum+1),desc="preparing relation matrix for every user……"):
        time_seq_each_user = np.zeros([maxlen],dtype=np.int32)
        index = maxlen-1
        #maxlen-1是因为下标从0开始
        for item_time in reversed(user_train[user][:-1]):
            time_seq_each_user[index]=item_time[1]
            index=index-1
            if index==-1:
                break
        all_relation_matrix[user]=cul_re_pos_matrix(time_seq_each_user,time_span)
    return  all_relation_matrix

#一次整个batch的取样操作
#需要取样的有user,user对应的seq和time_seq，time_matrix,pos_seq,neg_seq
#pos_seq对应的比原序列左移一格

def sample(user_train,user_num,item_num,batch_size,maxlen,relation_matrix, result_queue, SEED):
    #一次对于user的取样操作
    def one_sample(user):
        item_seq=np.zeros([maxlen],dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos_seq=np.zeros([maxlen],dtype=np.int32)
        neg_seq = np.zeros([maxlen], dtype=np.int32)
        pos_item = user_train[user][-1][0]
        time_matrix=relation_matrix[user]

        index=maxlen-1
        item_had = set(map(lambda x: x[0],user_train[user]))

        for item_time in reversed(user_train[user][:-1]):
            item_seq[index] = item_time[0]
            time_seq[index]=item_time[1]
            pos_seq[index]=pos_item
            if pos_item!=0:
                neg_seq[index]=random_neq_item(1,item_num,item_had)
            pos_item=item_time[0]
            index=index-1
            if index==-1:
                break
        return (user,item_seq,time_seq,time_matrix,pos_seq,neg_seq)
    #我还是不太能理解seed的作用 是我自己传进去的？那从什么时候开始生效呢
    #所以这里的seed其实是函数的第一句话啊 所以这个函数一旦调用seed就开始生效 后面的随机全部可以复现
    np.random.seed(SEED)
    #现在开始不断产生一个batch_size大小的sample 用于给模型不断输入（好神奇
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1,user_num+1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, user_num + 1)
            #如果交互次数太小 即用于训练的数据只有一组 无法构成前后关系 直接重新选择
            #好奇怪 我记得之前不是把有三个的训练数据 一个train一个valid一个test吗 这里为什么嫌弃它？？
            one_batch.append(one_sample(user))
        #为什么想拿到（user1，user2……） （seq1，seq2，……）这种构造而不是拿到（user1，seq1，……）这种构造呢
        result_queue.put(zip(*one_batch))

#这个类用于同步进行取样和训练过程
#User[user]={[item,timestap]……}
class data_sampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10,n_workers=1):
        #这里又生成一个随机数？？那种子的作用是？
        self.result_queue = Queue(maxsize=n_workers*10)
        self.processors=[]
        for i in range(n_workers):
            p=Process(target=sample,args=(User,
                                          usernum,
                                          itemnum,
                                          batch_size,
                                          maxlen,
                                          relation_matrix,
                                          self.result_queue,
                                          np.random.randint(2e9)
                                          ))
            self.processors.append(p)
            self.processors[-1].daemon = True
            self.processors[-1].start()
            #子进程————采样启动

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

#把时间全部变为绝对时间-》相对时间 方便后面绝对时间和相对时间的映射
#clean and sort的时候运用这个将绝对时间映射到相对时间
#clean and sort的时候还把相对时间映射为了个性化的相对时间
#所以relation_matrix 是在所有处理结束之后 将[time1,time2……]这些本来已经是个性化相对时间转换为相对时间矩阵
def timeslice(time_set):
    Min=min(time_set)
    time_map = dict()
    #为什么要这么多进制转换？？
    for time in time_set:
        time_map[time] = int(round(float(time-Min)))
    return time_map

#User是长度为usernum 每个user长度为各自的item，timestamp的总和 是一个字典
#做user和item到连续数字的映射
def clean_sort(User,time_map):
    user_set = set()
    item_set = set()
    pre_user = dict()

    #这里只是给所有的item和user一个集合方便后面的映射
    for user,item_times in User.items():
        user_set.add(user)
        pre_user[user]=item_times
        for item_time in item_times:
            item_set.add(item_time[0])

    #开始映射
    user_map = dict()
    item_map = dict()
    for u,user in enumerate(user_set):
        user_map[user]=u+1
    for i,item in enumerate(item_set):
        item_map[item]=i+1

    sort_by_time_users=dict()
    for user,item_times in pre_user.items():
        sort_by_time_users[user]=sorted(item_times,key=lambda x:x[1])

    user_ref = dict()
    for user,item_times in sort_by_time_users.items():
        number_of_user = user_map[user]
        user_ref[number_of_user]=list(map(lambda x: [item_map[x[0]], time_map[x[1]]], item_times))
        #主要是这个公式真的很好用

    #个性化的相对时间
    personal_time_max=set()
    #这是做什么用的？一定要有这个max吗
    # 先处理个性化的相对时间 （文章中怎么没提到啊）再将相对时间离散化
    for user,item_times in user_ref.items():
        time_list = list(map(lambda x:x[1],item_times))
        time_re = set()
        #这里不对啊？？？为什么是相邻两个的时间啊
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i]!=0:
                time_re.add(time_list[i+1]-time_list[i])
        if len(time_re)==0:
            time_scale=1
        else:
            time_scale=min(time_re)
        time_min=min(time_list)
        user_ref[user]=list(map(lambda x:[x[0], int(round((x[1]-time_min)/time_scale)+1)],item_times))
        personal_time_max.add(max(set(map(lambda x: x[1], user_ref[user]))))

    user_num=len(user_set)
    item_num = len(item_set)
    time_max = max(personal_time_max)
    return user_ref,user_num,item_num,time_max

#从文件中直接提取最原始的数据 分为训练集 验证集 测试集
def data_partition(file_name):
    user_num = 0
    item_num=0
    time_num = 0
    User= defaultdict(list)
    user_train = dict()
    user_valid = dict()
    user_test = dict()
    time_set = set()

    print("pre-processing data……")
    f=open('data/%s.txt' % file_name, 'r')

    user_interact_num=defaultdict(int)
    item_interact_num = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u=int(u)
        i=int(i)

        #记一遍数字
        user_interact_num[u]+=1
        item_interact_num[i]+=1
    f.close()
    f=open('data/%s.txt' % file_name, 'r')
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp=float(timestamp)
        #这里为什么是float
        if user_interact_num[u]<5 or item_interact_num[i]<5:
            continue
        time_set.add(timestamp)
        User[u].append([i,timestamp])
    f.close()

    time_map=timeslice(time_set)
    User,user_num,item_num,time_num = clean_sort(User,time_map)
    #clean_sort里传入的User是没有整理过的 仅仅只做到的u和[item,timestamp]对应
    #而出来的User将[item,stamp]按照时间步排序了
    for user in User:
        interat_num=len(User[user])
        #明明小于5的都没记进来 为何呢
        if interat_num<3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print("Done!")
    return [user_train,user_valid,user_test,user_num,item_num,time_num]

def evaluate_valid(model,dataset,args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    #独立拷贝 不让函数的改变影响到原数据
    NDCG=0.0
    HT = 0.0
    for_cul_usernum=0.0

    if usernum>10000:
        users=random.sample(range(1,usernum+1),10000)
    else:
        users=range(1,usernum+1)
    for user in users:
        if len(train[user])<1 or len(valid[user])<1:
            continue
        else:
           seq = np.zeros([args.maxlen], dtype=np.int32)
           time_seq=np.zeros([args.maxlen],dtype=np.int32)
           index=args.maxlen-1

           for item_times in reversed(train[user]):
               seq[index]=item_times[0]
               time_seq[index]=item_times[1]
               index-=1
               if index==-1:
                   break
           item_looked=set(map(lambda x: x[0],train[user]))
           item_looked.add(valid[user][0][0])
           item_looked.add(test[user][0][0])
           item_looked.add(0)
           item_for_evaluate=[valid[user][0][0]]
           for _ in range(100):
               s = np.random.randint(1,itemnum+1)
               while s in item_looked:
                   s = np.random.randint(1,itemnum+1)
               item_for_evaluate.append(s)
           time_matrix=cul_re_pos_matrix(time_seq,args.time_span)
           #原实现太怪了
           #这里实际上就是把原本的单层数组多一个batch 使其能match predict函数的要求
           predictions = -model.predict(np.array([user]),
                                      np.array([seq]),
                                      np.array([time_matrix]),
                                      np.array(item_for_evaluate))
           prediction=predictions[0]

           rank = prediction.argsort().argsort()[0].item()
           for_cul_usernum+=1

           if rank<10:
               NDCG += (1 / np.log2(rank + 2))
               HT += 1
           if for_cul_usernum % 100 == 0:
               print('.', end='')
               sys.stdout.flush()
    return NDCG/for_cul_usernum,HT/for_cul_usernum

def evaluate(model,dataset,args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    #独立拷贝 不让函数的改变影响到原数据
    NDCG=0.0
    HT = 0.0
    for_cul_usernum=0.0

    if usernum>10000:
        users=random.sample(range(1,usernum+1),10000)
    else:
        users=range(1,usernum+1)
    for user in users:
        if len(train[user])<1 or len(valid[user])<1:
            continue
        else:
           seq = np.zeros([args.maxlen], dtype=np.int32)
           time_seq=np.zeros([args.maxlen],dtype=np.int32)
           index=args.maxlen-1

           seq[index]=valid[user][0][0]
           time_seq[index]=valid[user][0][1]
           index-=1

           for item_times in reversed(train[user]):
               seq[index]=item_times[0]
               time_seq[index]=item_times[1]
               index-=1
               if index==-1:
                   break
           item_looked=set(map(lambda x: x[0],train[user]))
           item_looked.add(valid[user][0][0])
           item_looked.add(test[user][0][0])
           item_looked.add(0)
           item_for_evaluate=[test[user][0][0]]
           for _ in range(100):
               s = np.random.randint(1,itemnum+1)
               while s in item_looked:
                   s = np.random.randint(1,itemnum+1)
               item_for_evaluate.append(s)

           time_matrix=cul_re_pos_matrix(time_seq,args.time_span)
           #原实现太怪了
           #这里实际上就是把原本的单层数组多一个batch 使其能match predict函数的要求
           predictions = -model.predict(np.array([user]),
                                      np.array([seq]),
                                      np.array([time_matrix]),
                                      np.array(item_for_evaluate))
           prediction=predictions[0]

           rank = prediction.argsort().argsort()[0].item()
           for_cul_usernum+=1

           if rank<10:
               NDCG += (1 / np.log2(rank + 2))
               HT += 1
           if for_cul_usernum % 100 == 0:
               print('.', end='')
               sys.stdout.flush()
    return NDCG/for_cul_usernum,HT/for_cul_usernum

