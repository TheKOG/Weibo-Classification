import json
import os
import sys
input_json=[]
if __name__=='__main__':
    try:
        arg=sys.argv[1].strip('/').strip("\\").strip('/')+'/'
    except:
        arg='input_csv/'
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