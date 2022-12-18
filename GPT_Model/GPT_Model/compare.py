import numpy as np
import torch
import mindspore as ms

from src.mindspore_version.load_pretrain_model import load_GPT_base as load_ms
from src.pytorch_version.load_pretrain_model import pre_defined_GPT_config
from src.pytorch_version.huggingface_gpt import GPT2Model
from tokenizer import get_sst_tokenizer
from logger import logger


def compare_model_weights(ms_GPT, py_GPT):
    """
    比较Mindspore和Pytorch版的GPT参数
    :param ms_GPT: Mindspore GPT_Model
    :param py_GPT: Pytorch GPT_Model
    :return: 误差值
    """
    ms_dict = list(ms_GPT.get_parameters())
    pyt_dict = list(py_GPT.named_parameters())

    diff = 0
    for x, y in zip(ms_dict, pyt_dict):
        print(x.name, "<--->", y[0])
        ms_weights = x.value().asnumpy()
        py_weights = y[1].detach().numpy()
        delta = np.sum(np.abs(ms_weights - py_weights))
        print("误差为: ", delta)
        diff += np.sum(delta)
    return diff


def compare_vocab_embedding(sentence, ms_GPT, py_GPT):
    # 获取提词器
    tokenizer = get_sst_tokenizer()
    inputs = tokenizer(sentence, return_tensors="pt")
    # 补足128位
    for key in inputs:
        inputs[key] = torch.concat([inputs[key], torch.zeros(1, 128 - inputs[key].shape[1], dtype=inputs[key].dtype)],
                                   1)
    # 打开测试模式，去除dropout的影响
    py_GPT.eval()
    py_output = py_GPT(**inputs).pool_output.detach().numpy()

    inputs["input_mask"] = inputs.get("attention_mask")
    inputs.pop("attention_mask")
    # 把torch变成mindspore
    for key in inputs:
        inputs[key] = ms.Tensor.from_numpy(inputs[key].numpy()).squeeze()
    sequence_output, pooled_output, embedding_tables = ms_GPT.construct(**inputs)
    ms_output = pooled_output.asnumpy()
    logger.critical("开始SST数据集embedding表示测试！！！")
    logger.info(f"语句输入为{sentence}")
    logger.critical("两个GPT的输出分别为：")
    logger.info("py->{}".format(py_output))
    logger.info("ms->{}".format(ms_output))

    logger.critical("embedding误差为{}".format(np.sum(np.abs((ms_output - py_output)))))
    return