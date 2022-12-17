from .loader_mindspore import set_encoder
from .gpt_mindspore import GPT2Model


def expand_GPT(org_GPT, target_GPT_config, method):
    """
    :param org_GPT: 预训练好的小GPT
    :param target_GPT_config: 大GPT的参数日志
    :param method: 扩展策略
    :return: 大GPT
    """
    if target_GPT_config.size_per_head is not None:
        assert target_GPT_config.size_per_head == org_GPT.size_per_head

    new_GPT = GPT2Model(target_GPT_config, is_training=True)
    encoder = []
    # 找到Encoder块
    modules = org_GPT.name_cells()
    for key in modules.keys():
        if "encoder" in key:
            encoder.append(modules.get(key))

    modules = new_GPT.name_cells()
    for key in modules.keys():
        if "encoder" in key:
            encoder.append(modules.get(key))
    set_encoder(new_GPT, org_GPT, encoder[0], encoder[1], org_hidden_size=org_GPT.hidden_size,
                target_hidden_size=new_GPT.hidden_size,
                method=method,
                new_num_layers=new_GPT.num_hidden_layers)
    return new_GPT
