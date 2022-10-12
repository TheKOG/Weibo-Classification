#!/usr/bin/env python
from random import randint
from random import random
from time import time
import numpy as np
import re
class PseudoDocTM(object):
    def __init__(self,P,K,iter,innerStep,saveStep,alpha1,alpha2,beta,inputPath,outputPath):
        self.K1=P
        self.K2=K
        self.niters=iter
        self.innerSteps= innerStep
        self.saveStep =saveStep
        self.alpha1=alpha1
        self.alpha2= alpha2
        self.beta = beta
        self.inputPath=inputPath
        self.outputPath=outputPath
        self.w2i={}
        self.i2w={}
        self.docs=[]
    

    def loadTxts(self,txtPath):
        reader = open(file=txtPath,mode='r',encoding='utf-8')
        try:
            line = reader.readline()
            while line:
                #print("fkpps!")
                doc = []
                tokens = line.strip()
                tokens=filter(None,re.split(u'[^\u4e00-\u9fa5a-zA-Z]',tokens))
                for token in tokens:
                    #print(token)
                    if (not (token in self.w2i)):
                        self.w2i[token]=len(self.w2i)
                        self.i2w[self.w2i[token]]=token
                    doc.append(self.w2i[token])
                self.docs.append(doc)
                line = reader.readline()
            reader.close()
        except:
            pass
        #文档数量
        self.M = len(self.docs)
        #语料词的数量
        self.V = len(self.w2i)
        return

    def initModel(self):

        self.mp = [0 for i in range(self.K1)]

        self.npk = [[0 for i in range(self.K2)] for j in range(self.K1)]
        self.npkSum = [0 for i in range(self.K1)]

        self.nkw = [[0 for i in range(self.V)] for j in range(self.K2)]
        self.nkwSum = [0 for i in range(self.K2)]

        self.zAssigns_1 = [0 for i in range(self.M)]#文档所属的伪文档
        self.zAssigns_2 = [[] for i in range(self.M)]#文档每个单词所属的主题

        for m in range(self.M):
            #文档单词的数量
            N = len(self.docs[m])
            #初始化
            self.zAssigns_2[m] = [0 for i in range(N)]
            #随机分配文档所属的伪文档
            z1 = randint(0,self.K1-1)
            self.zAssigns_1[m] = z1

            self.mp[z1] +=1 #伪文档对应的文本数量增加
            #对每个单词随机分配主题
            for n in range(N):
                w = self.docs[m][n]
                z2 = randint(0,self.K2-1)

                self.npk[z1][z2] +=1
                self.npkSum[z1] +=1

                self.nkw[z2][w] +=1
                self.nkwSum[z2] +=1

                self.zAssigns_2[m][n] = z2
    
    
    #抽取文档所属的伪文档
    def sampleZ1(self,m):
        z1 = self.zAssigns_1[m]  #获取文档所属的伪文档
        N = len(self.docs[m]) #获取文档单词的数量

        self.mp[z1] -=1 #移除该文档，伪文档z1对应的单词数量减少

        k2Count ={}
        for n in range(N): #循环文档的每个单词
            z2 = self.zAssigns_2[m][n] #获取单词的主题分配
            if (z2 in k2Count): #计算每个主题包含该文档单词的总数量
                k2Count[z2]+=1
            else:
                k2Count[z2]=1
            self.npk[z1][z2] -=1
            self.npkSum[z1] -=1

        k2Alpha2 = self.K2 * self.alpha2   #分母的K*alpha

        pTable = [0.0 for i in range(self.K1)]
        #循环每个伪文档
        for k in range(self.K1):
            expectTM = 1.0
            index = 0
            #这里要计算单词的频次，进行连乘
            for z2 in k2Count:
                c = k2Count[z2]
                for i in range(c):
                    expectTM *= (self.npk[k][z2] + self.alpha2 + i) / (k2Alpha2 + self.npkSum[k] + index)
                    index +=1
            #基于公式计算概率
            pTable[k] = (self.mp[k] + self.alpha1) / (self.M + self.K1 * self.alpha1) * expectTM
        #轮盘赌选择
        for k in range(1,self.K1):#这里注意k=1开始，不能k=0
            pTable[k] += pTable[k-1]

        r = random() * pTable[self.K1-1]
        z1=0
        for k in range(self.K1):
            if (pTable[k] > r):
                z1 = k
                break
        #基于轮盘赌选择的伪文档，重新统计
        self.mp[z1] +=1
        for n in range(N):
            z2 = self.zAssigns_2[m][n]
            self.npk[z1][z2] +=1
            self.npkSum[z1] +=1

        self.zAssigns_1[m] = z1

    
    #抽取文档m第n个单词的主题
    def sampleZ2(self,m,n):

        z1 = self.zAssigns_1[m] #获取文档所属的伪文档
        z2 = self.zAssigns_2[m][n] #获取文档m第n个所属的主题
        w = self.docs[m][n] #获取单词编号

        self.npk[z1][z2] -=1  #统计伪文档z1、主题z2生成的单词数量
        self.npkSum[z1] -=1 #伪文档z1对应的总单词数量
        self.nkw[z2][w] -=1 #主题z2对应的单词w的数量
        self.nkwSum[z2] -=1 #主题z2中所有单词的数量

        VBeta = self.V * self.beta #分母中的V*beta
        k2Alpha2 = self.K2 * self.alpha2 #分母中的 K*alpha

        pTable = [0.0 for i in range(self.K2)]
        #基于公式计算-----这里和公式有差异,公式应该按照这里写，及主题词分母应该按照前面的表达
        for k in range(self.K2):
            pTable[k] = (self.npk[z1][k] + self.alpha2) / (self.npkSum[z1] + k2Alpha2) *(self.nkw[k][w] + self.beta) / (self.nkwSum[k] + VBeta)
        #轮盘赌选择
        for k in range(1,self.K2):
            pTable[k] += pTable[k-1]

        r = random() * pTable[self.K2-1]

        for k in range(self.K2):
            if (pTable[k] > r):
                z2 = k
                break
        #重新统计相关词频
        self.npk[z1][z2] +=1
        self.npkSum[z1] +=1
        self.nkw[z2][w] +=1
        self.nkwSum[z2] +=1

        self.zAssigns_2[m][n] = z2
        
    def estimate(self):
        start = 0
        for iter in range(self.niters):
            start = time()
            print("PAM4ST Iteration: " + str(iter) + " ...")
            if(iter%self.saveStep==0 and iter!=0 and iter!=self.niters-1):
                self.storeResult(iter)
            #对每篇文档循环，将文档分配到伪文档
            for i in range(self.innerSteps):
                for m in range(self.M):
                    self.sampleZ1(m)
            #对每篇文档进行循环，抽取每个单词所属的主题
            for i in range(self.innerSteps):
                for m in range(self.M):
                    N = len(self.docs[m])
                    for n in range(N):
                        self.sampleZ2(m, n)
            print("cost time:"+str(time()-start))
        return

    def computeThetaP(self):
        theta = [[0.0 for i in range(self.K2)] for j in range(self.K1)]
        for k1 in range(self.K1):
            for k2 in range(self.K2):
                theta[k1][k2] = (self.npk[k1][k2] + self.alpha2) / (self.npkSum[k1] + self.K2*self.alpha2)
        return theta
    
    def saveThetaP(self,path):
        writer = open(file=path,mode='w',encoding='utf-8')
        theta = self.computeThetaP()
        for k1 in range(self.K1):
            for k2 in range(self.K2):
                writer.write(str(theta[k1][k2])+" ")
            writer.write('\n')
        writer.flush()
        writer.close()
    
    def saveZAssigns1(self,path):
        writer = open(file=path,mode='w',encoding='utf-8')
        for m in range(self.M):
            writer.write(str(self.zAssigns_1[m])+"\n")
        writer.flush()
        writer.close()
    
    #计算主题词分布
    def computePhi(self):
        phi = [[0.0 for i in range(self.V)] for j in range(self.K2)]
        for k in range(self.K2):
            for v in range(self.V):
                phi[k][v] = (self.nkw[k][v] + self.beta) / (self.nkwSum[k] + self.V*self.beta)
        return phi
    
    def printTopics(self,path,top_n):
        writer=open(file=path,mode='w',encoding='utf-8')
        phi=self.computePhi()
        for k in range(self.K2):
            writer.write('Topic '+str(k)+":\n")
            n=min(len(phi[k]),top_n)
            mx=len(phi[k])
            lis=np.array(phi[k]).argsort()
            for i in range(n):
                index=lis[mx-1-i]
                writer.write(str(self.i2w[index])+" "+str(phi[k][index])+'\n')
            continue
        writer.close()
    
    def savePhi(self,path):
        writer=open(file=path,mode='w',encoding='utf-8')
        phi = self.computePhi()
        K = len(phi)
        assert K > 0
        V = len(phi[0])
        try:
            for k in range(K):
                for v in range(V):
                    writer.write(str(phi[k][v])+" ");
                writer.write("\n")
            writer.flush()
            writer.close()
        except:
            pass
    
    def saveWordmap(self,path):
        writer=open(file=path,mode='w',encoding='utf-8')
        try:
            for word in self.w2i:
                writer.write(word + " " + str(self.w2i[word]) + "\n")
            writer.flush()
            writer.close()
        except:
            pass

    def saveAssign(self,path):
        writer=open(file=path,mode='w',encoding='utf-8')
        try:
            for i in range(len(self.zAssigns_2)):
                for j in range(len(self.zAssigns_2[i])):
                    writer.write(str(self.docs[i][j])+":"+str(self.zAssigns_2[i][j])+" ")
                writer.write("\n")
            writer.flush()
            writer.close()
        except:
            pass
    
    def printModel(self):
        print("\tK1 :"+str(self.K1)+
            "\tK2 :"+str(self.K2)+
            "\tniters :"+str(self.niters)+
            "\tinnerSteps :"+str(self.innerSteps)+
            "\tsaveStep :"+str(self.saveStep) +
            "\talpha1 :"+str(self.alpha1)+
            "\talpha2 :"+str(self.alpha2)+
            "\tbeta :"+str(self.beta)+
            "\tinputPath :"+self.inputPath+
            "\toutputPath :"+self.outputPath)
    
    def convert_zassigns_to_arrays_theta(self):
        self.ndk = [[0 for i in range(self.K2)] for j in range(self.M)]
        self.ndkSum = [0 for i in range(self.M)]
        #print('fuck pps!')
        for m in range(self.M):
            for n in range(len(self.docs[m])):
                self.ndk[m][self.zAssigns_2[m][n]]+=1
                self.ndkSum[m]+=1
    
    def computeTheta(self):
        self.convert_zassigns_to_arrays_theta()
        theta = [[0.0 for i in range(self.K2)] for j in range(self.M)]
        for m in range(self.M):
            for k in range(self.K2):
                theta[m][k] = (self.ndk[m][k] + self.alpha2) / (self.ndkSum[m] + self.K2 * self.alpha2)
        return theta
    
    def saveTheta(self,path):
        writer=open(file=path,mode='w',encoding='utf-8')
        theta = self.computeTheta()
        try:
            for m in range(self.M):
                for k in range(self.K2):
                    writer.write(str(theta[m][k])+" ")
                writer.write("\n")
            writer.flush()
            writer.close()
            #print('pps!')
        except:
            pass

    def storeResult(self,times):
        appendString="final"
        if(times!=0):
            appendString =str(times)
        try:
            self.printTopics(self.outputPath+"/model-"+appendString+".twords",10)
            self.saveWordmap(self.outputPath+"/wordmap.txt")
            self.savePhi(self.outputPath+"/model-"+appendString+".phi")
            self.saveAssign(self.outputPath+"/model-"+appendString+".tassign")
            self.saveTheta(self.outputPath+"/model-"+appendString+".theta")
            self.saveThetaP(self.outputPath+"/model-"+appendString+".thetap")
            self.saveZAssigns1(self.outputPath+"/model-"+appendString+".assign1")
        except:
            pass

    def run(self):
        self.printModel()
        self.loadTxts(self.inputPath)#加载语料
        self.initModel()#初始化模型
        self.estimate()#估计
        self.storeResult(0)#保存结果

if __name__=='__main__':
    ptm=PseudoDocTM(4,2,10,20,20,0.1,0.01,0.1,"test.txt","output")
    ptm.run()