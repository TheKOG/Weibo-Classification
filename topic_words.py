from sklearn.feature_extraction.text import CountVectorizer

def init_topics(input_pth='data/cutted.txt'):
    docs=[]
    with open(input_pth,'r',encoding='utf-8') as f:
        line=f.readline()
        while(line):
            docs.append(line)
            line=f.readline()
        f.close()
    n_features = 2000 #提取2000个特征词语
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                    max_features=n_features,
                                    stop_words='english',
                                    max_df = 0.5,
                                    min_df = 10)
    tf = tf_vectorizer.fit_transform(docs)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    with open('data/topic_words.txt','w',encoding='utf-8') as f:
        for i in range(n_features):
            f.write(tf_feature_names[i]+" {0}".format(i)+'\n')
            f.flush()
        f.close()