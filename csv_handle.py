import csv
import os
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

def Load_csv(dir_path="csvs",out_path='data/csv.txt'):
    dirs=os.listdir(dir_path)
    out=open(out_path,'w',encoding='utf-8')
    for dir in dirs:
        path='csvs/'+dir+'/'+dir+'.csv'
        if(is_UTF8(path)):
            ecd='utf-8'
        else:
            ecd='gbk'
        with open(path,encoding = ecd) as f:
            f_csv=csv.reader(f,skipinitialspace=True)
            # print(dir)
            for row in f_csv:
                out.write(row[4]+'\n')
                out.flush()
    out.close()