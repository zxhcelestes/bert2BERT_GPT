import os.path

import numpy

from src.GPT_Model.pytorch_version.load_pretrain_model import load_GPT_base as ld_py, enlarge as enlarge_py, \
    pre_defined_GPT_config as config_py
from src.GPT_Model.mindspore_version.load_pretrain_model import load_GPT_base as ld_ms, enlarge as enlarge_ms, \
    pre_defined_GPT_config as config_ms
from get_path import root


class PytorchEnlarger:

    # ckpt_path = "../../../ckpt/gpt_small.ckpt"
    # save_path = "../../../output/gpt_small2base_aki.pth"
    # model = load_gpt_base(ckpt_path, "gpt_small", load_gitee_ckpt=False, specify_prefix=None)
    # new_model = enlarge(model, pre_defined_gpt_config("gpt_base"), "AKI", save_path)

    # ckpt_path = "../ckpt/gpt_base.ckpt"
    # save_path = "../output/gpt_base2large_aki.pth"
    # model = load_gpt_base(ckpt_path, "gpt_base", load_gitee_ckpt=False, specify_prefix=None)
    # new_model = enlarge(model, pre_defined_gpt_config("gpt_large"), "AKI", save_path)
    @staticmethod
    def gpt_large2xl_aki():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_large", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_xl"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_large2xl_fpi():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_large", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_xl"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_aki():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_medium", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_fpi():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_medium", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_large"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_aki():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_base", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_medium"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_fpi():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_base", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_medium"), "FPI", save_path)
            return new_model


class MindsporeEnlarger:

    # save_path = "../../../output/gpt_small2base_aki.ckpt"
    # pre_train_model = load_gpt_base(path="../../../ckpt/gpt_small.ckpt", kind="gpt_small", load_gitee_ckpt=False)
    # new_model = enlarge(pre_train_model, pre_defined_gpt_config("gpt_base"), "AKI", save_path)
    @staticmethod
    def gpt_large2xl_aki():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_large", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_xl"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_large2xl_fpi():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_large", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_xl"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_aki():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_medium", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_fpi():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_medium", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_large"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_aki():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="gpt_base", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_medium"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_fpi():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="gpt_base", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_medium"), "FPI", save_path)
            return new_model
