from .GPT_config import GPT2Config
from .huggingface_gpt import GPT2Model
from .loader import set_encoder


def expand_GPT(org_GPT, target_GPT_config, method="FPI"):
    """

    :param org_GPT: 预训练好的小GPT
    :param target_GPT_config: 大GPT的参数日志
    :param method: 扩展策略
    :return: 大GPT
    """
    if target_GPT_config.size_per_head is not None:
        assert target_GPT_config.size_per_head == org_GPT.config.size_per_head

    new_GPT = GPT2Model(target_GPT_config)
    encoder = []
    # 找到Encoder块
    modules = org_GPT.named_children()
    for key, data in modules:
        if "encoder" in key:
            encoder.append(data)
    modules = new_GPT.named_children()
    for key, data in modules:
        if "encoder" in key:
            encoder.append(data)
    org_enc = encoder[0]
    new_enc = encoder[1]
    set_encoder(new_GPT, org_GPT, org_enc, new_enc, org_hidden_size=org_GPT.config.hidden_size,
                target_hidden_size=new_GPT.config.hidden_size,
                method=method,
                new_num_layers=new_GPT.config.num_hidden_layers)

    return new_GPT


if __name__ == '__main__':
    org_model = GPT2Model(GPT2Config(num_hidden_layers=12, hidden_size=120))
    new_model = GPT2Model(GPT2Config(vocab_size=28996, num_hidden_layers=24, hidden_size=240))
    # 预训练GPT
    # org_model = GPTModel.from_pretrained("GPT-base-cased")
    expand_GPT(org_GPT=org_model, target_GPT_config=GPT2Config(num_hidden_layers=24, hidden_size=240))
