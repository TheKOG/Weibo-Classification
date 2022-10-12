import java.io.*;
import java.util.*;
import java.util.Map.Entry;

class PseudoDocTM{

    public int K1 = 1000;  //设置伪文档数量
    public int K2 = 100; //主题数量

    public int M;
    public int V;

    public double alpha1 = 0.1;
    public double alpha2 = 0.1;

    public double beta = 0.01;

    public int mp[]; //分配到每个伪文档文档的数量

    public int npk[][];  //伪文档l由主题k生成的单词数量
    public int npkSum[];  //伪文档对应的单词总数

    public int nkw[][]; //主题k对应的单词w的数量
    public int nkwSum[]; //主题k对应的单词总数

    public int zAssigns_1[];  //文档分配伪文档
    public int zAssigns_2[][]; //文档单词分配主题

    public int niters = 200; 
    public int saveStep = 1000; 
    public String inputPath="input.txt";
    public String outputPath="";

    public int innerSteps = 10;

    public List<List<Integer>> docs = new ArrayList<List<Integer>>(); //文档表示
    public HashMap<String, Integer> w2i = new HashMap<String, Integer>(); //词的编号
    public HashMap<Integer, String> i2w = new HashMap<Integer, String>(); //编号转化为词


    public PseudoDocTM(int P,int K,int iter,int innerStep,int saveStep,double alpha1,double alpha2,double beta,String inputPath,String outputPath){
        this.K1=P;
        this.K2=K;
        this.niters=iter;
        this.innerSteps= innerStep;
        this.saveStep =saveStep;
        this.alpha1=alpha1;
        this.alpha2= alpha2;
        this.beta = beta;
        this.inputPath=inputPath;
        this.outputPath=outputPath;
    }
    //加载语料
    public void loadTxts(String txtPath) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(txtPath), "utf-8"));

        String line;
        try {
            line = reader.readLine();
            while (line != null) {
                List<Integer> doc = new ArrayList<Integer>();

                String[] tokens = line.trim().split("[^\u4e00-\u9fa5]");
                for (String token : tokens) {
                    if(token=="")continue;
                    if (!w2i.containsKey(token)) {
                        w2i.put(token, w2i.size());
                        i2w.put(w2i.get(token), token);
                    }
                    doc.add(w2i.get(token));
                }
                docs.add(doc);
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        //文档数量
        M = docs.size();
        //语料词的数量
        V = w2i.size();

        return;
    }
    //初始化模型
    public void initModel() {

        mp = new int[K1];

        npk = new int[K1][K2];
        npkSum = new int[K1];

        nkw = new int[K2][V];
        nkwSum = new int[K2];

        zAssigns_1 = new int[M]; //文档所属的伪文档
        zAssigns_2 = new int[M][]; //文档每个单词所属的主题

        for (int m = 0; m < M; m++) {
            //文档单词的数量
            int N = docs.get(m).size();
            //初始化
            zAssigns_2[m] = new int[N];
            //随机分配文档所属的伪文档
            int z1 = (int) Math.floor(Math.random()*K1);
            zAssigns_1[m] = z1;

            mp[z1] ++; //伪文档对应的文本数量增加
            //对每个单词随机分配主题
            for (int n = 0; n != N; n++) {
                int w = docs.get(m).get(n);
                int z2 = (int) Math.floor(Math.random()*K2);

                npk[z1][z2] ++;
                npkSum[z1] ++;

                nkw[z2][w] ++;
                nkwSum[z2] ++;

                zAssigns_2[m][n] = z2;
            }
        }
    }
    //抽取文档所属的伪文档
    public void sampleZ1(int m) {
        int z1 = zAssigns_1[m];  //获取文档所属的伪文档
        int N = docs.get(m).size(); //获取文档单词的数量

        mp[z1] --; //移除该文档，伪文档z1对应的单词数量减少

        Map<Integer, Integer> k2Count = new HashMap<Integer, Integer>();
        for (int n = 0; n != N; n++){ //循环文档的每个单词
            int z2 = zAssigns_2[m][n]; //获取单词的主题分配
            if (k2Count.containsKey(z2)) { //计算每个主题包含该文档单词的总数量
                k2Count.put(z2, k2Count.get(z2)+1);
            } else {
                k2Count.put(z2, 1);
            }

            npk[z1][z2] --;
            npkSum[z1] --;
        }

        double k2Alpha2 = K2 * alpha2;   //分母的K*alpha

        double[] pTable = new double[K1];
        //循环每个伪文档
        for (int k = 0; k != K1; k++) {
            double expectTM = 1.0;
            int index = 0;
            //这里要计算单词的频次，进行连乘
            for (int z2 : k2Count.keySet()) {
                int c = k2Count.get(z2);
                for (int i = 0; i != c; i++) {
                    expectTM *= (npk[k][z2] + alpha2 + i) / (k2Alpha2 + npkSum[k] + index);
                    index ++;
                }
            }
            //基于公式计算概率
            pTable[k] = (mp[k] + alpha1) / (M + K1 * alpha1) * expectTM;
        }
        //轮盘赌选择
        for (int k = 1; k != K1; k++) { //这里注意k=1开始，不能k=0
            pTable[k] += pTable[k-1];
        }

        double r = Math.random() * pTable[K1-1];

        for (int k = 0; k != K1; k++) {
            if (pTable[k] > r) {
                z1 = k;
                break;
            }
        }
        //基于轮盘赌选择的伪文档，重新统计
        mp[z1] ++;
        for (int n =0; n != N; n++) {
            int z2 = zAssigns_2[m][n];
            npk[z1][z2] ++;
            npkSum[z1] ++;
        }

        zAssigns_1[m] = z1;
    }
    //抽取文档m第n个单词的主题
    public void sampleZ2(int m, int n) {

        int z1 = zAssigns_1[m]; //获取文档所属的伪文档
        int z2 = zAssigns_2[m][n]; //获取文档m第n个所属的主题
        int w = docs.get(m).get(n); //获取单词编号

        npk[z1][z2] --;  //统计伪文档z1、主题z2生成的单词数量
        npkSum[z1] --; //伪文档z1对应的总单词数量
        nkw[z2][w] --; //主题z2对应的单词w的数量
        nkwSum[z2] --; //主题z2中所有单词的数量

        double VBeta = V * beta; //分母中的V*beta
        double k2Alpha2 = K2 * alpha2; //分母中的 K*alpha

        double[] pTable = new double[K2];
        //基于公式计算-----这里和公式有差异,公式应该按照这里写，及主题词分母应该按照前面的表达
        for (int k = 0; k != K2; k++) {
            pTable[k] = (npk[z1][k] + alpha2) / (npkSum[z1] + k2Alpha2) *
                    (nkw[k][w] + beta) / (nkwSum[k] + VBeta);
        }
        //轮盘赌选择
        for (int k = 1; k != K2; k++) {
            pTable[k] += pTable[k-1];
        }

        double r = Math.random() * pTable[K2-1];

        for (int k = 0; k != K2; k++) {
            if (pTable[k] > r) {
                z2 = k;
                break;
            }
        }
        //重新统计相关词频
        npk[z1][z2] ++;
        npkSum[z1] ++;
        nkw[z2][w] ++;
        nkwSum[z2] ++;

        zAssigns_2[m][n] = z2;
        return;
    }

    public void estimate() {
        long start = 0;
        for (int iter = 0; iter != niters; iter++) {
            start = System.currentTimeMillis();
            System.out.println("PAM4ST Iteration: " + iter + " ...");
            if(iter%this.saveStep==0&&iter!=0&&iter!=this.niters-1){
                this.storeResult(iter);
            }
            //对每篇文档循环，将文档分配到伪文档
            for (int i = 0; i != innerSteps; i++) {
                for (int m = 0; m != M; m++) {
                    this.sampleZ1(m);
                }
            }
            //对每篇文档进行循环，抽取每个单词所属的主题
            for (int i = 0; i != innerSteps; i++) {
                for (int m = 0; m != M; m++) {
                    int N = docs.get(m).size();
                    for (int n = 0; n != N; n++) {
                        sampleZ2(m, n);
                    }
                }
            }
            System.out.println("cost time:"+(System.currentTimeMillis()-start));
        }
        return;
    }
    //计算伪文档的主题分布---相当于LDA的文档主题分布
    public double[][] computeThetaP() {
        double[][] theta = new double[K1][K2];
        for (int k1 = 0; k1 != K1; k1++) {
            for (int k2 = 0; k2 != K2; k2++) {
                theta[k1][k2] = (npk[k1][k2] + alpha2) / (npkSum[k1] + K2*alpha2);
            }
        }
        return theta;
    }

    public void saveThetaP(String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(path)));
        double[][] theta = this.computeThetaP();
        for (int k1 = 0; k1 != K1; k1++) {
            for (int k2 = 0; k2 != K2; k2++) {
                writer.append(theta[k1][k2]+" ");
            }
            writer.newLine();
        }
        writer.flush();
        writer.close();
    }

    public void saveZAssigns1(String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(path)));

        for (int m = 0; m != M; m++) {
            writer.append(zAssigns_1[m]+"\n");
        }

        writer.flush();
        writer.close();
    }
    //计算主题词分布
    public double[][] computePhi() {
        double[][] phi = new double[K2][V];
        for (int k = 0; k != K2; k++) {
            for (int v = 0; v != V; v++) {
                phi[k][v] = (nkw[k][v] + beta) / (nkwSum[k] + V*beta);
            }
        }
        return phi;
    }
    //排序算法
    public ArrayList<List<Entry<String, Double>>> sortedTopicWords(
            double[][] phi, int T) {
        ArrayList<List<Entry<String, Double>>> res = new ArrayList<List<Entry<String, Double>>>();
        for (int k = 0; k != T; k++) {
            HashMap<String, Double> term2weight = new HashMap<String, Double>();
            for (String term : w2i.keySet())
                term2weight.put(term, phi[k][w2i.get(term)]);

            List<Entry<String, Double>> pairs = new ArrayList<Entry<String, Double>>(
                    term2weight.entrySet());
            Collections.sort(pairs, new Comparator<Entry<String, Double>>() {
                public int compare(Entry<String, Double> o1,
                        Entry<String, Double> o2) {
                    return (o2.getValue().compareTo(o1.getValue()));
                }
            });
            res.add(pairs);
        }
        return res;
    }


    public void printTopics(String path,int top_n) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));
        double[][] phi = computePhi();
        ArrayList<List<Entry<String, Double>>> pairsList = this
                .sortedTopicWords(phi, K2);
        for (int k = 0; k != K2; k++) {
            writer.write("Topic " + k + ":\n");
            for (int i = 0; i != top_n; i++) {
                //System.out.printf("fk:%d\n",i);
                writer.write(pairsList.get(k).get(i).getKey() + " "
                        + pairsList.get(k).get(i).getValue()+"\n");
                //System.out.printf("pps:%d\n",i);
            }
        }
        writer.close();
    }

    public void savePhi(String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));

        double[][] phi = computePhi();
        int K = phi.length;
        assert K > 0;
        int V = phi[0].length;

        try {
            for (int k = 0; k != K; k++) {
                for (int v = 0; v != V; v++) {
                    writer.append(phi[k][v]+" ");
                }
                writer.append("\n");
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return;
    }

    public void saveWordmap(String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));
        try {
            for (String word : w2i.keySet()){
                writer.append(word + " " + w2i.get(word) + "\n");
            }

            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return;
    }

    public void saveAssign(String path)throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));
        try {
            for(int i=0;i<zAssigns_2.length;i++){
                for(int j=0;j<zAssigns_2[i].length;j++){
                    writer.write(docs.get(i).get(j)+":"+zAssigns_2[i][j]+" ");
                }
                writer.write("\n");
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return;
    }
    public void printModel(){
        System.out.println("\tK1 :"+this.K1+
                "\tK2 :"+this.K2+
                "\tniters :"+this.niters+
                "\tinnerSteps :"+this.innerSteps+
                "\tsaveStep :"+this.saveStep +
                "\talpha1 :"+this.alpha1+
                "\talpha2 :"+this.alpha2+
                "\tbeta :"+this.beta +
                "\tinputPath :"+this.inputPath+
                "\toutputPath :"+this.outputPath);
    }

    int[][] ndk;
    int[] ndkSum;

    public void convert_zassigns_to_arrays_theta(){
        ndk = new int[M][K2];
        ndkSum = new int[M];

        for (int m = 0; m != M; m++) {
            for (int n = 0; n != docs.get(m).size(); n++) {
                ndk[m][zAssigns_2[m][n]] ++;
                ndkSum[m] ++;
            }
        }
    }
    //计算文档主题分布
    public double[][] computeTheta() {
        convert_zassigns_to_arrays_theta();
        double[][] theta = new double[M][K2];
        for (int m = 0; m != M; m++) {
            for (int k = 0; k != K2; k++) {
                theta[m][k] = (ndk[m][k] + alpha2) / (ndkSum[m] + K2 * alpha2);
            }
        }
        return theta;
    }

    public void saveTheta(String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "utf-8"));

        double[][] theta = computeTheta();
        try {
            for (int m = 0; m != M; m++) {
                for (int k = 0; k != K2; k++) {
                    writer.append(theta[m][k]+" ");
                }
                writer.append("\n");
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return;
    }

    public void storeResult(int times){
        String appendString="final";
        if(times!=0){
            appendString =times+"";
        }
        try {
            this.printTopics(outputPath+"/model-"+appendString+".twords",10);
            this.saveWordmap(outputPath+"/wordmap.txt");
            this.savePhi(outputPath+"/model-"+appendString+".phi");
            this.saveAssign(outputPath+"/model-"+appendString+".tassign");
            this.saveTheta(outputPath+"/model-"+appendString+".theta");
            this.saveThetaP(outputPath+"/model-"+appendString+".thetap");
            this.saveZAssigns1(outputPath+"/model-"+appendString+".assign1");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    public void run() throws IOException {
        printModel();
        this.loadTxts(inputPath);//加载语料
        this.initModel(); //初始化模型
        this.estimate(); //估计
        this.storeResult(0); //保存结果

    }

    public static void main(String args[]) throws IOException {
        int args_[]=new int[3];
        args_[0]=500;
        args_[1]=60;
        args_[2]=30;
        int pt=0;
        for(String arg:args){
            args_[pt]=Integer.parseInt(arg);
            pt++;
        }
        PseudoDocTM ptm=new PseudoDocTM(args_[0],args_[1],args_[2],50,20,0.1,0.01,0.1,"input/cutted.txt","ptm_result");
        ptm.run();
    }
}