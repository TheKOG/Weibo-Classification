from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import os
import jieba
import jieba.posseg as psg
import re
from tqdm import tqdm
import csv_handle
import cut
import topic_words
def load_stop_words(stop_file='data/stopwords.txt'):
    try:
        stopword_list = open(stop_file,encoding ='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_dict = {}
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_dict[line]=1
    return stop_dict

def chinese_word_cut(mytext,stop_dict=[]):
    ban_list = ['m','q','r','p','c','u','w','uj','ul','d']
    word_list = []
    #jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)
        if(len(word)<2):
            continue
        # if(word=='减肥'):
        #     print('fkpps!')
        if seg_word.flag not in ban_list and word not in stop_dict:
            word_list.append(word)
    return (" ").join(word_list)

doc_name=[]

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

def get_docs():
    dirs=os.listdir("csvs")
    jieba.initialize()
    load_stop_words()
    docs=[]
    for i,dir in enumerate(tqdm(dirs,desc='Loading')):
        doc_name.append(dir)
        # print("psg:"+dir.strip('%23'))
        path='csvs/'+dir+'/'+dir+'.csv'
        if(is_UTF8(path)):
            ecd='utf-8'
        else:
            ecd='gbk'
        with open(path,encoding = ecd) as f:
            f_csv=csv.reader(f,skipinitialspace=True)
            text=[]
            for row in f_csv:
                text.append(chinese_word_cut(row[4]))
            docs.append(' '.join(text))
    return docs

if __name__=='__main__':
    csv_handle.Load_csv()
    cut.Create_Cutted(out_pth='data/cutted.txt')
    topic_words.init_topics(input_pth='data/cutted.txt')
    voc={}
    rev={}
    with open('data/topic_words.txt',encoding='utf-8') as f:
        line=f.readline()
        while line:
            v,id=line.strip('\n').split(' ')
            id=int(id)
            voc[v]=id
            rev[id]=v
            line=f.readline()
        f.close()
    vectorizer=TfidfVectorizer(vocabulary=voc,sublinear_tf=True)
    docs=get_docs()
    n_docs=len(docs)
    # print(n_docs)
    tf=vectorizer.fit_transform(docs)
    mat=tf.toarray()
    with open('data/topics.name','w',encoding='utf-8') as f:
        for i,topic in enumerate(doc_name):
            f.write(topic.strip('%23')+" {0}".format(i)+'\n')
        f.close()
    with open('data/topics.tfmat','w',encoding='utf-8') as f:
        for i in range(n_docs):
            sum=0
            if i!=0:
                f.write('\n')
            for j in range(len(voc)):
                f.write(str(mat[i][j])+" ")
                sum+=mat[i][j]
            if(sum==0):
                print('fuck '+doc_name[i])
        f.close()