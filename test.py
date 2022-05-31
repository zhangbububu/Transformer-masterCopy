"""
Created bt tz on 2020/11/12 
"""

__author__ = 'tz'
import torch
from nltk import word_tokenize
from torch.autograd import Variable
from utils import subsequent_mask
from setting import MAX_LENGTH, DEVICE
import numpy as np
from utils import bleu_candidate

MOD1 , MOD2 = 1,2
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    print('src = ',src)
    print('src_mask = ',src_mask)
    memory = model.encode(src, src_mask)

    print('mamory.maen = ',memory.mean())
    # memory.shape(batch,sentence_len,d_model)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        dec_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))

        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys



def eva_one_sentence(data,model,str):

    str = ["BOS"] + word_tokenize(str.lower())  + ["EOS"]
    print(str)
    str = [data.en_word_dict.get(i,0) for i in str]
    print(str)
    # return

    # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
    src = torch.from_numpy(np.array(str)).long().to(DEVICE)
    # 增加一维
    src = src.unsqueeze(0)
    # 设置attention mask
    src_mask = (src != 0).unsqueeze(-2)
    # 用训练好的模型进行decode预测
    out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
    # 初始化一个用于存放模型翻译结果句子单词的列表
    translation = []
    # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）

    attention_weights = []

    for j in range(1, out.size(1)):
        # 获取当前下标的输出字符
        sym = data.cn_index_dict[out[0, j].item()]
        # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
        if sym != 'EOS':
            translation.append(sym)
        # 否则终止遍历
        else:
            break
    # 打印模型翻译输出的中文句子结果
    print("translation: %s" % " ".join(translation))
    bleu_candidate(" ".join(translation))
    return " ".join(translation)

def evaluate(data, model):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """
    # 梯度清零

    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.dev_en)):
            # 打印待翻译的英文句子
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            # 打印对应的中文句子答案
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print(cn_sent)

            # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果句子单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文句子结果
            print("translation: %s" % " ".join(translation))
            bleu_candidate(" ".join(translation))
            return " ".join(translation)


def evaluate_test(data, model):
    evaluate(data, model)

def init(mode=0,str=None):
    from setting import TRAIN_FILE, DEV_FILE
    from train import model
    from data_pre import PrepareData
    model.load_state_dict(torch.load('save/model.pt', map_location=torch.device('cpu')))
    data = PrepareData(TRAIN_FILE, DEV_FILE)

    if(mode ==0):
        evaluate_test(data, model)
    else:
        return eva_one_sentence(data,model,str)


if __name__ == '__main__':
    init(1,'hi')
