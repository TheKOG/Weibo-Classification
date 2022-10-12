from sklearn.feature_extraction.text import TfidfVectorizer
import ImgCap
import cut
import os
import json
from tqdm import tqdm
import csv
from scipy import spatial
wordmap={}
rev_wm={}
def LoadWordMap(path='data/topic_words.txt'):
    with open(path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            a,b=line.strip('\n').split(' ')
            wordmap[a]=b
            rev_wm[b]=a
    return

doc_json=[]

imgtxt={}

docs=[]
doc_id=[]

def create_imgs_input(img_dir):
    imgs=os.listdir(img_dir)
    # print(imgs)
    for i,img_pth in enumerate(tqdm(imgs,desc="Captioning Images in "+img_dir)):
        id=img_pth.strip('.jpg').split('-')[0]
        text=ImgCap.estimate(img_dir+'/'+img_pth)
        if id not in imgtxt:
            imgtxt[id]=[]
        imgtxt[id].append(text)

def is_UTF8(path):
    with open(path,encoding = "utf-8") as f:
        try:
            f_csv=csv.reader(f,skipinitialspace=True)
            for row in f_csv:
                text=row
                f.close()
                return True
        except:
            f.close()
            return False

def create_docs_input(csv_pth,f,topic='unknown'):
    if(is_UTF8(csv_pth)):
        ecd='utf-8'
    else:
        ecd='gbk'
    with open(csv_pth,encoding = ecd) as fr:
        f_csv=csv.reader(fr,skipinitialspace=True)
        # print(dir)
        for row in f_csv:
            text=row[4]
            id=row[0]
            if id in imgtxt:
                text+=" "+" ".join(imgtxt[id])
            docs.append(text)
            doc_id.append(id)
            doc_json.append(topic)
            f.write(text.strip("\n")+"\n")
            f.flush()

def Init_Data(input_json='input/input.json',out_json='input/docs.json',out_txt='input/docs.txt'):
    ImgCap.Load('./ImgCap/model.pth','./ImgCap/wordmap.json')
    with open(input_json,'r',encoding='utf-8') as f:
        datas=json.load(f)
        f.close()
    with open(out_txt,'w',encoding='utf-8') as f:
        for data in datas:
            if 'images' in data:
                create_imgs_input(img_dir=data['images'])
            if 'docs' in data:
                try:
                    topic=data['topic']
                except:
                    topic='unknown'
                create_docs_input(csv_pth=data['docs'],f=f,topic=topic)
        f.close()
    
    with open(out_json,'w',encoding='utf-8') as f:
        json.dump(doc_json,f)
        f.close()
    
    cut.Create_Cutted(data_pth='input/docs.txt',out_pth="input/cutted.txt")

d2p={}
pdocs=[]
p2t={}
topic_name={}
topic_tf=[]
classified={}

def Validate(psum=100,logpth="output/log.txt",outpth="output/result.json"):
    pdocs=["" for i in range(psum)]
    with open('ptm_result/model-final.assign1') as f:
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
    n_doc=len(docs)
    for i in range(n_doc):
        real=doc_json[i]
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
        f.write('Topic:{0} Precision={1},Recall={1},Accuracy={2},F1 score={3}\n'.format(topic,percision[topic],recall[topic],accuracy[topic],F1[topic]))

    percision_/=tot
    recall_/=tot
    accuracy_/=tot
    F1_/=tot
    f.write('Average: Precision={1},Recall={1},Accuracy={2},F1 score={3}\n'.format(percision_,recall_,accuracy_,F1_))
    
    with open(outpth,"w",encoding='utf-8') as f:
        json.dump(classified,f)
        f.close()


if __name__=='__main__':
    Init_Data()
    psum=1000
    ksum=100
    niters=100
    os.system("java ptm.java {0} {1} {2}".format(psum,ksum,niters))
    Validate(psum)