import mindspore as ms
import torch

from .GPT_config import GPT2Config
from .huggingface_gpt import GPT2Model
from .model_expander import expand_GPT


def pre_defined_GPT_config(name):
    if name == "gpt_base":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=768,
                          n_layer=12,
                          n_head=12,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)

    elif name == "gpt_medium":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=1024,
                          n_layer=24,
                          n_head=16,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)

    elif name == "gpt_large":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=1280,
                          n_layer=36,
                          n_head=20,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)

    elif name == "gpt_xl":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=1600,
                          n_layer=48,
                          n_head=25,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)
    else:
        raise Exception("????????????????????????")


def trans_ms_2_pytorch(param):
    # ???numpy
    weights = param.value().asnumpy()
    return [param.name, torch.tensor(weights, dtype=torch.float32, requires_grad=True)]


def pytorch_load_ckpt(path):
    ckpt_dict = ms.load_checkpoint(path)
    new_dict = dict()
    for key in ckpt_dict.keys():
        new_dict[key] = trans_ms_2_pytorch(ckpt_dict.get(key))
    return new_dict


def load_GPT_base(path, kind, filter_prefix=None, specify_prefix=None, load_gitee_ckpt=False):
    params = ms.load_checkpoint(path, filter_prefix=filter_prefix,
                                specify_prefix=specify_prefix)
    params_dict = dict()
    if load_gitee_ckpt:
        for key in params.keys():
            # ?????????
            new_key = ".".join(key.split(".")[2:])
            params_dict[new_key] = trans_ms_2_pytorch(params.get(key))
    else:
        params_dict = dict()
        for key in params.keys():
            params_dict[key] = trans_ms_2_pytorch(params.get(key))

    GPT_config = pre_defined_GPT_config(kind)
    model = GPT2Model(GPT_config)

    pyt_dict = model.state_dict()
    # print(len(pyt_dict.keys()))
    # quit()
    org_keys = list(params_dict.keys())
    new_keys = list()
    tmp=0
    for name,param in model.named_parameters():
        new_keys.append(name)
    for x, y in zip(org_keys, new_keys):
        # print(x, "     ", y)
        # print(params_dict.get(x)[1].shape, "     ", pyt_dict.get(y).shape)
        pyt_dict[y] = params_dict.get(x)[1]
    # pyt_dict['embeddings.position_ids'] = position_ids
    model.load_state_dict(state_dict=pyt_dict, strict=True)
    return model


def enlarge(model, target_config, method, save_path):
    new_model = expand_GPT(model, target_config, method)
    torch.save(new_model.state_dict(), save_path)
    return new_model


if __name__ == '__main__':
    ckpt_path = "../../ckpt/GPT_base.ckpt"
    save_path = "../../output/gpt_base2medium_fpi.pth"
    model = load_GPT_base(ckpt_path, "gpt_base", load_gitee_ckpt=False)
    new_model = enlarge(model, pre_defined_GPT_config("gpt_medium"), "FPI", save_path)
    for idx, layer in enumerate(new_model.named_parameters()):
        print("-" * 40)
        print(layer[0], "-->", layer[1])
        print("-" * 40)

    # ckpt_path = "../ckpt/GPT_small.ckpt"
    # save_path = "../output/GPT_small2base_aki.pth"
    # model = load_GPT_base(ckpt_path, "GPT_small", load_gitee_ckpt=False, specify_prefix=None)
    # new_model = enlarge(model, pre_defined_GPT_config("GPT_base"), "AKI", save_path)
    # for idx, layer in enumerate(new_model.named_parameters()):
    #     print("-" * 40)
    #     print(layer[0], "-->", layer[1])
    #     print("-" * 40)