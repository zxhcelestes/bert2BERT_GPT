import argparse

import torch

from compare_on_gpt import compare_model_weights, compare_vocab_embedding
from src.GPT_Model.mindspore_version.load_pretrain_model import load_GPT_base as load_ms
from src.GPT_Model.pytorch_version.load_pretrain_model import pre_defined_GPT_config
from src.GPT_Model.pytorch_version.huggingface_gpt import GPT2Model
from enlarger_on_gpt import PytorchEnlarger, MindsporeEnlarger
from logger import logger
import random
import time


class Comparator:
    sentence = "Hello, my dog is cute."
    timecost = []

    def test_base2medium_fpi(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.gpt_base2medium_fpi()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.gpt_base2medium_fpi()
        self.timecost.append(time.time() - start)
        logger.critical(f"Pytorch扩展用时{self.timecost[-2]}秒，MindSpore扩展用时{self.timecost[-1]}秒")

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
        start = time.time()
        PytorchEnlarger.gpt_medium2large_fpi()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.gpt_medium2large_fpi()
        self.timecost.append(time.time() - start)
        logger.critical(f"Pytorch扩展用时{self.timecost[-2]}秒，MindSpore扩展用时{self.timecost[-1]}秒")

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

    def test_large2xl_fpi(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.gpt_large2xl_fpi()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.gpt_large2xl_fpi()
        self.timecost.append(time.time() - start)
        logger.critical(f"Pytorch扩展用时{self.timecost[-2]}秒，MindSpore扩展用时{self.timecost[-1]}秒")

        # pytorch
        large_config = pre_defined_GPT_config("gpt_xl")
        pytorch_model = GPT2Model(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/gpt_large2xl_fpi.pth"))

        # mindspore
        ms_model = load_ms(path="output/gpt_large2xl_fpi.ckpt", specify_prefix=None, kind="gpt_xl")
        logger.info("-" * 40)
        logger.info("large2xl_fpi模型参数误差为:{} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_large2xl_aki(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.gpt_large2xl_aki()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.gpt_large2xl_aki()
        self.timecost.append(time.time() - start)
        logger.critical(f"Pytorch扩展用时{self.timecost[-2]}秒，MindSpore扩展用时{self.timecost[-1]}秒")

        # pytorch
        large_config = pre_defined_GPT_config("gpt_xl")
        pytorch_model = GPT2Model(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/gpt_large2xl_aki.pth"))

        # mindspore
        ms_model = load_ms(path="output/gpt_large2xl_aki.ckpt", specify_prefix=None, kind="gpt_xl")
        logger.info("-" * 40)
        logger.info("large2xl_fpi模型参数误差为:{} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_medium2large_aki(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.gpt_medium2large_aki()
        self.timecost.append(time.time()-start)

        random.seed(10)
        start=time.time()
        MindsporeEnlarger.gpt_medium2large_aki()
        self.timecost.append(time.time()-start)
        logger.critical(f"Pytorch扩展用时{self.timecost[-2]}秒，MindSpore扩展用时{self.timecost[-1]}秒")

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
        start=time.time()
        PytorchEnlarger.gpt_base2medium_aki()
        self.timecost.append(time.time()-start)

        random.seed(10)
        start=time.time()
        MindsporeEnlarger.gpt_base2medium_aki()
        self.timecost.append(time.time()-start)
        logger.critical(f"Pytorch扩展用时{self.timecost[-2]}秒，MindSpore扩展用时{self.timecost[-1]}秒")

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
    parser.add_argument('--task_type', type=str, default="", choices=["base2medium", "medium2large","large2xl"],
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
    elif args.task_type == "large2xl":
        if args.method == "fpi":
            comparator.test_large2xl_fpi()
        elif args.method == "aki":
            comparator.test_large2xl_aki()
        else:
            raise Exception("invalid args")
    else:
        raise Exception("invalid args")


if __name__ == '__main__':
    run_eval()
