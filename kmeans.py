import torch
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from scipy import spatial

topic={}
class Kmeans(object):
    def __init__(self,k,n_voc):
        self.k=k
        self.n=n_voc
        self.topics=torch.rand(k,n_voc)
        self.topics.requires_grad_(True)
        # self.topics.retain_grad()
        # print(self.topics.is_leaf)

    def forward(self,x):
        ret=torch.Tensor(x.size()[0])
        # print(x.size()[0])
        for i in range(x.size()[0]):
            row=x[i]
            out=((self.topics-row)**2).sum(1).sqrt()
            # print(out.argmin())
            topic[i]=out.argmin().item()
            ret[i]=out.min()
        # print(ret)
        return ret.mean()
    
    def Learn(self,input,lr=0.02,step=200):
        input=torch.Tensor(input)
        for i,i in enumerate(tqdm(range(step),desc="Learning")):
            y=self.forward(input)
            y.backward(retain_graph=True)
            # print(y)
            # print(self.topics.is_leaf)
            with torch.autograd.no_grad():
                # print(self.topics.grad)
                self.topics-=self.topics.grad*lr
                self.topics.grad.zero_()

def Load():
    global docs
    global doc_topic
    global doc_id
    global voc
    global doc_mat
    voc={}
    rev={}
    with open('data/topic_words.txt',encoding='utf-8') as f:
        line=f.readline()
        while line:
            v,id=line.strip('\n').strip().split(' ')
            id=int(id)
            voc[v]=id
            rev[id]=v
            line=f.readline()
        f.close()
    
    with open('input/docs.json','r',encoding='utf-8') as f:
        doc_json=json.load(f)
    doc_topic,doc_id=doc_json['topic'],doc_json['id']
    
    docs=[]

    with open('input/cutted.txt','r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            if line!='':
                docs.append(line)
    
    vectorizer=TfidfVectorizer(vocabulary=voc,sublinear_tf=True)
    doc_mat=vectorizer.fit_transform(docs)
    doc_mat=doc_mat.toarray()

d2p={}
pdocs=[]
p2t={}
topic_name={}
topic_tf=[]

def Validate(psum=100,logpth="output/kmeans_log.txt"):
    pdocs=["" for i in range(psum)]
    with open('kmeans_result/topics.txt') as f:
        lines=f.readlines()
        for i,line in enumerate(lines):
            if line=='':
                break
            val=int(line.strip('\n').strip())
            d2p[i]=val
            if pdocs[val]=='':
                pdocs[val]=docs[i].strip()
            else:
                pdocs[val]+=" "+docs[i].strip()
    vectorizer=TfidfVectorizer(vocabulary=voc,sublinear_tf=True)
    
    with open('data/topics.name',encoding='utf-8') as f:
        line=f.readline()
        while line:
            tmp=line.strip('\n').strip().split(" ")
            topic_name[int(tmp[1])]=tmp[0]
            line=f.readline()
        f.close()
        
    with open('data/topics.tfmat',encoding='utf-8') as f:
        line=f.readline()
        while line:
            tmp=line.strip('\n').strip().split(" ")
            topic_tf.append([float(ele) for ele in tmp])
            line=f.readline()
        f.close()
    
    tf=vectorizer.fit_transform(pdocs)
    tf=tf.toarray()
    topic_n=len(topic_tf)
    for i in range(psum):
        mx=-1
        for j in range(topic_n):
            cosi=1-spatial.distance.cosine(tf[i],topic_tf[j])
            if cosi>mx:
                mx=cosi
                p2t[i]=topic_name[j]
    
    TP={}
    FP={}
    TN={}
    FN={}
    n_doc=len(docs)
    for i in range(n_doc):
        real=doc_topic[i]
        esti=p2t[d2p[i]]
        if real=='unknown':
            continue
        for ele in [real,esti]:
            for dic in [TP,FP,TN,FN]:
                if ele not in dic:
                    dic[ele]=0
        # print("("+real+","+esti+")")
        if real==esti:
            TP[real]+=1
        else:
            FN[real]+=1
            FP[esti]+=1

    percision={}
    recall={}
    accuracy={}
    F1={}
    tot=0
    percision_=0
    recall_=0
    accuracy_=0
    F1_=0
    f=open(logpth,'w',encoding='utf-8')
    for topic in TP:
        tot+=1
        TN[topic]=n_doc-TP[topic]-FP[topic]-FN[topic]
        # print('Topic:{0} TP={1},FP={1},FN={2},TN={3}\n'.format(topic,TP[topic],FP[topic],FN[topic],TN[topic]))
        percision[topic]=TP[topic]/(TP[topic]+FP[topic]+0.01)
        recall[topic]=TP[topic]/(TP[topic]+FN[topic]+0.01)
        accuracy[topic]=(TP[topic]+TN[topic])/n_doc
        F1[topic]=(2*percision[topic]*recall[topic])/(percision[topic]+recall[topic]+0.01)
        percision_+=percision[topic]
        recall_+=recall[topic]
        accuracy_+=accuracy[topic]
        F1_+=F1[topic]
        f.write('Topic:{0} Precision={1},Recall={2},Accuracy={3},F1 score={4}\n'.format(topic,percision[topic],recall[topic],accuracy[topic],F1[topic]))

    percision_=percision_/tot
    recall_=recall_/tot
    accuracy_=accuracy_/tot
    F1_=F1_/tot
    f.write('Average: Precision={0},Recall={1},Accuracy={2},F1 score={3}\n'.format(percision_,recall_,accuracy_,F1_))

def k_means(psum=100):
    Load()
    kmeans=Kmeans(psum,len(voc))
    kmeans.Learn(doc_mat)
    with open("kmeans_result/topics.txt",'w',encoding='utf-8') as f:
        for i in range(len(docs)):
            f.write(str(topic[i])+'\n')
            f.flush()
        f.close()
    Validate(psum)