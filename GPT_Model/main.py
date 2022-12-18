import argparse

import torch

from compare import compare_model_weights, compare_vocab_embedding
from src.mindspore_version.load_pretrain_model import load_GPT_base as load_ms
from src.pytorch_version.load_pretrain_model import pre_defined_GPT_config
from src.pytorch_version.huggingface_gpt import GPT2Model
from enlarger import PytorchEnlarger, MindsporeEnlarger
from logger import logger
import random


class Comparator:
    sentence = "Hello, my dog is cute."

    def test_base2medium_fpi(self):
        random.seed(10)
        PytorchEnlarger.gpt_base2medium_fpi()
        random.seed(10)
        MindsporeEnlarger.gpt_base2medium_fpi()
        # pytorch
        large_config = pre_defined_GPT_config("gpt_medium")
        pytorch_model = GPT2Model(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/gpt_base2medium_fpi.pth"))

        # mindspore
        ms_model = load_ms(path="output/gpt_base2medium_fpi.ckpt", specify_prefix=None, kind="gpt_medium")
        logger.info("-" * 40)
        logger.info("base2medium_fpi模型参数误差为:{} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_medium2large_fpi(self):
        random.seed(10)
        PytorchEnlarger.gpt_medium2large_fpi()
        random.seed(10)
        MindsporeEnlarger.gpt_medium2large_fpi()
        # pytorch
        large_config = pre_defined_GPT_config("gpt_large")
        pytorch_model = GPT2Model(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/gpt_medium2large_fpi.pth"))

        # mindspore
        ms_model = load_ms(path="output/gpt_medium2large_fpi.ckpt", specify_prefix=None, kind="gpt_large")
        logger.info("-" * 40)
        logger.info("medium2large_fpi模型参数误差为:{} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_medium2large_aki(self):
        random.seed(10)
        PytorchEnlarger.gpt_medium2large_aki()
        random.seed(10)
        MindsporeEnlarger.gpt_medium2large_aki()
        # pytorch
        large_config = pre_defined_GPT_config("gpt_large")
        pytorch_model = GPT2Model(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/gpt_medium2large_aki.pth"))

        # mindspore
        ms_model = load_ms(path="output/gpt_medium2large_aki.ckpt", specify_prefix=None, kind="gpt_large")
        logger.info("-" * 40)
        logger.info("medium2large_aki模型参数误差为:{} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_base2medium_aki(self):
        random.seed(10)
        PytorchEnlarger.gpt_base2medium_aki()
        random.seed(10)
        MindsporeEnlarger.gpt_base2medium_aki()
        # pytorch
        large_config = pre_defined_GPT_config("gpt_medium")
        pytorch_model = GPT2Model(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/gpt_base2medium_aki.pth"))

        # mindspore
        ms_model = load_ms(path="output/gpt_base2medium_aki.ckpt", specify_prefix=None, kind="gpt_medium")
        logger.info("-" * 40)
        logger.info("base2medium_aki模型参数误差为:{} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)


def run_eval():
    """ evaluate scripts """
    parser = argparse.ArgumentParser(description="gpt_Model expansion")
    parser.add_argument('--task_type', type=str, default="", choices=["base2medium", "medium2large"],
                        help="expansion type.")
    parser.add_argument('--method', type=str, default="fpi", choices=["fpi", "aki"], help="Expansion strategy.")
    comparator = Comparator()
    args = parser.parse_args()
    if args.task_type == "base2medium":
        if args.method == "fpi":
            comparator.test_base2medium_fpi()
        elif args.method == "aki":
            comparator.test_base2medium_aki()
        else:
            raise Exception("invalid args")
    elif args.task_type == "medium2large":
        if args.method == "fpi":
            comparator.test_medium2large_fpi()
        elif args.method == "aki":
            comparator.test_medium2large_aki()
        else:
            raise Exception("invalid args")
    else:
        raise Exception("invalid args")


if __name__ == '__main__':
    c= Comparator()
    c.test_base2medium_fpi()

