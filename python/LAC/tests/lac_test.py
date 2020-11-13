from LAC import LAC
import time
from tqdm import tqdm
"""
熟悉代码用的，使用后放入pre_WordDict
"""

def fun_seg():
    # 装载分词模型
    lac = LAC("models_general/seg_model",mode='seg')

    # 单个样本输入，输入为Unicode编码的字符串
    text = u"LAC是个优秀的分词工具"
    seg_result = lac.run(text)
    print(seg_result)

    # 批量样本输入, 输入为多个句子组成的list，平均速率会更快
    texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
    seg_result = lac.run(texts)
    print(seg_result)

def fun_add_word():
    lac = LAC(model_path='tmp/lac/my_lac_model', mode='seg')
    lac.add_word('红红 火火', sep=None)
    seg_result = lac.run("他这一生红红火火了一把")
    print(seg_result)

def run():
    # 选择使用分词模型
    lac = LAC(model_path='/home/work/zhouchengjie/tmp/lac/models_general/lac_model')
    # lac = LAC(model_path='tmp/lac/my_lac_model')
    result = lac.run('百度是一家很好的公司')
    print(result)
    return 0

    train_file = '/home/disk0/huangdingbang/lac_data/wd_data.tsv'
    test_file = None

    # 使用自己训练好的模型
    zero = time.time()
    train_data = []
    sence = []
    with open(train_file, 'r', encoding='utf-8')as f:
       for a, line in tqdm(enumerate(f)):
            if a % 50 != 0:
                words = line.strip("\n").split("\t")[0]  # 训练数据处理过程
                words = [word for i, word in enumerate(words) if i%2==0]
                sence += words
            else:
                train_data.append(''.join(sence))
                sence = []
            if a == 50000:
                break
            
    start = time.time()
    print('pre continue {} s'.format(start-zero))
    print('start testing...')
    for i in range(len(train_data)):
        seg_result = lac.run(train_data[i])
    end = time.time()
    print('pro continue {} s'.format(end-start))

def train():
    my_lac = LAC(model_path='/home/work/zhouchengjie/tmp/lac/my_lac_model')

    test_file =  "/home/work/zhouchengjie/test/cws_data/hkcantcor_test.txt"
    train_file = 'tmp/lac/python/pre_WordDict/train.txt'
    use_file = '/home/disk0/huangdingbang/lac_data/wd_data.tsv'
    # test_file = None
    my_lac.train(model_save_dir='/home/work/zhouchengjie/tmp/lac/test_lac_model/',train_data=train_file, test_data=None)
    seg_result = my_lac.run("他这一生红红火火了一把")
    print(seg_result)

    my_lac = LAC(model_path='/home/work/zhouchengjie/tmp/lac/test_lac_model/')
    my_lac.add_word('红红 火火', sep=None)
    seg_result = my_lac.run("他这一生红红火火了一把")
    print(seg_result)

# fun_add_word()
run()
# train()

