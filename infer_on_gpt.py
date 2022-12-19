import argparse

import mindspore as ms
import torch

from logger import logger
from src.GPT_Model.mindspore_version.load_pretrain_model import load_GPT_base as load_ms
from tokenizer import get_sst_tokenizer


def _infer_mindspore(sentence, ms_gpt_path, kind):
    """
    mindspore的gpt推理
    :param sentence: 语句
    :param ms_gpt_path: mindspore的gpt的ckpt文件
    :param kind: gpt类型
    :return: embedding
    """

    ms_gpt = load_ms(ms_gpt_path, kind)
    # 获取提词器
    tokenizer = get_sst_tokenizer()
    inputs = tokenizer(sentence, return_tensors="pt")
    # 补足128位
    for key in inputs:
        inputs[key] = torch.concat([inputs[key], torch.zeros(1, 128 - inputs[key].shape[1], dtype=inputs[key].dtype)],
                                   1)
    inputs["input_mask"] = inputs.get("attention_mask")
    inputs.pop("attention_mask")
    # 把torch变成mindspore
    for key in inputs:
        inputs[key] = ms.Tensor.from_numpy(inputs[key].numpy()).squeeze()
    sequence_output, pooled_output, embedding_tables = ms_gpt.construct(**inputs)
    ms_output = pooled_output.asnumpy()
    logger.info(f"语句输入为{sentence}")
    logger.info("ms->{}".format(ms_output))
    return ms_output


def infer_ms():
    parser = argparse.ArgumentParser(description="vocab embedding infer")
    parser.add_argument('--sentence', type=str, default="",
                        help="sentence")
    parser.add_argument('--ms_gpt_path', type=str, help="gpt ckpt file")
    parser.add_argument('--kind', type=str, choices=["gpt_base", "gpt_large", "gpt_small"], help="gpt kind")
    args = parser.parse_args()
    print(args)
    _infer_mindspore(args.sentence, args.ms_gpt_path, args.kind)


if __name__ == '__main__':
    # _infer_mindspore("what do you think?", "./output/gpt_small2base_aki.ckpt", "gpt_base")
    infer_ms()
