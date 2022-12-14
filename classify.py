from sklearn.feature_extraction.text import TfidfVectorizer
import json
from scipy import spatial
from LDA import LDA
from kmeans import k_means
import os

d2p={}
pdocs=[]
p2t={}
topic_name={}
topic_tf=[]
classified={}
cutted=[]

def Validate(psum=500,logpth="output/ptm_log.txt",outpth="output/ptm_result.json"):
    pdocs=["" for i in range(psum)]
    with open('input/docs.json','r',encoding='utf-8') as f:
        doc_json=json.load(f)
    doc_topic,doc_id=doc_json['topic'],doc_json['id']
    print(len(doc_topic))
    print(len(doc_id))
    with open('input/cutted.txt','r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            if line!='':
                cutted.append(line)
    print(len(cutted))
    with open('ptm_result/model-final.assign1') as f:
        lines=f.readlines()
        for i,line in enumerate(lines):
            if line=='':
                break
            val=int(line.strip('\n').strip())
            # print("{0} {1} {2} {3} {4}".format(psum,val,i,len(docs),cutted[i]))
            d2p[i]=val
            if pdocs[val]=='':
                pdocs[val]=cutted[i].strip()
            else:
                pdocs[val]+=" "+cutted[i].strip()
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
    n_doc=len(cutted)
    for i in range(n_doc):
        real=doc_topic[i]
        esti=p2t[d2p[i]]
        if esti not in classified:
            classified[esti]=[]
        classified[esti].append(doc_id[i])
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

    percision_/=tot
    recall_/=tot
    accuracy_/=tot
    F1_/=tot
    f.write('Average: Precision={0},Recall={1},Accuracy={2},F1 score={3}\n'.format(percision_,recall_,accuracy_,F1_))
    
    with open(outpth,"w",encoding='utf-8') as f:
        json.dump(classified,f)
        f.close()


if __name__=='__main__':
    psum=500
    ksum=100
    niters=100
    os.system("java ptm.java {0} {1} {2}".format(psum,ksum,niters))
    Validate(psum)
    LDA(psum)
    k_means(psum=100)