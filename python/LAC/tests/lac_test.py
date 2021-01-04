from LAC import LAC
"""
本文件测试LAC的分词,词性标注及重要性功能
"""
os.environ['PYTHONIOENCODING'] = 'UTF-8'

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
    lac = LAC(model_path='models_general/seg_model', mode='seg')
    lac.add_word('红红 火火', sep=None)
    seg_result = lac.run("他这一生红红火火了一把")
    print(seg_result)

def run():
    # 选择使用lac模型
    lac = LAC(model_path='models_general/lac_model', mode='lac')
    result = lac.run('百度是一家很好的公司')
    print(result)

def rank():
    # 选择使用rank模型
    lac = LAC(model_path='models_general/rank_model', mode='rank')
    result = lac.run('百度是一家很好的公司')
    print(result)


