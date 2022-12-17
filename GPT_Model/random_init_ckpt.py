from src.mindspore_version.load_pretrain_model import pre_defined_GPT_config
from src.mindspore_version.gpt_mindspore import GPT2Model
import mindspore
from mindspore.common.initializer import initializer, Normal


def random_init_ckpt(path, kind):
    """
    随机生成kind型的gpt参数
    :param path: 保存地址
    :param kind: gpt规格，如gpt_base
    """
    gpt_config = pre_defined_GPT_config(kind)
    model = GPT2Model(gpt_config, is_training=True)
    dic = dict()
    for param in model.get_parameters():
        name = param.name
        shape = param.shape
        dic[name] = initializer(Normal(), shape, mindspore.float32)
    # 参数load进模型
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    mindspore.load_param_into_net(model, dic, strict_load=False)
    mindspore.save_checkpoint(model, path)


if __name__ == '__main__':
    path = "./ckpt/gpt_base"
    random_init_ckpt(path, "gpt_base")
