import jieba
import jieba.posseg as psg
import re
from tqdm import tqdm
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
        if seg_word.flag not in ban_list and word not in stop_dict:
            word_list.append(word)
    return (" ").join(word_list)

def Create_Cutted(data_pth='data/csv.txt',dic_file='data/dict.txt',stop_file='data/stopwords.txt',out_pth='data/cutted.txt'):
    out=open(out_pth,'w',encoding='utf-8')
    jieba.load_userdict(dic_file)
    jieba.initialize()
    stop_dict=load_stop_words(stop_file)
    with open(data_pth,'r',encoding='utf-8') as f:
        lines=f.readlines()
        for i,line in enumerate(tqdm(lines,"Tokenizing")):
            if(line==''):
                continue
            out.write(chinese_word_cut(line,stop_dict)+'\n')
            out.flush()
            line=f.readline()
    out.close()