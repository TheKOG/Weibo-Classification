import json
import os
import sys
from tqdm import tqdm
import csv
import ImgCap
import cut
import os

input_json=[]

doc_topic=[]

imgtxt={}

docs=[]
doc_id=[]

def create_imgs_input(img_dir):
    imgs=os.listdir(img_dir)
    # print(imgs)
    for i,img_pth in enumerate(tqdm(imgs,desc="Captioning Images in "+img_dir)):
        try:
            id=img_pth.strip('.jpg').split('-')[0]
            text=ImgCap.estimate(img_dir+'/'+img_pth)
            if id not in imgtxt:
                imgtxt[id]=[]
            imgtxt[id].append(text)
        except:
            pass

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
            docs.append(text.strip("\n"))
            doc_id.append(id)
            doc_topic.append(topic)

def Init_Data(input_json='input/input.json',out_json='input/docs.json'):
    ImgCap.Load('./ImgCap/model.pth','./ImgCap/wordmap.json')
    with open(input_json,'r',encoding='utf-8') as f:
        datas=json.load(f)
        f.close()
    for data in datas:
        if 'images' in data:
            create_imgs_input(img_dir=data['images'])
        if 'docs' in data:
            try:
                topic=data['topic']
            except:
                topic='unknown'
            create_docs_input(csv_pth=data['docs'],f=f,topic=topic)
    
    with open(out_json,'w',encoding='utf-8') as f:
        json.dump({'topic':doc_topic,'id':doc_id,'doc':docs},f)
        f.close()
    
    cut.Create_Cutted(data_pth='input/docs.json',out_pth="input/cutted.txt")

if __name__=='__main__':
    try:
        arg=sys.argv[1].strip('/').strip("\\").strip('/')+'/'
    except:
        arg='input_csv/'
    print("Loading from "+arg)
    dirs=os.listdir(arg)
    for dir in dirs:
        pth=arg+dir
        ls=os.listdir(pth)
        tmp={}
        if("images" in ls):
            tmp['images']=pth+'/images'
        tmp["docs"]=pth+'/'+dir+'.csv'
        tmp['topic']=dir.strip('%23')
        input_json.append(tmp)
    with open("input/input.json","w",encoding='utf-8') as f:
        json.dump(input_json,f)
    Init_Data()