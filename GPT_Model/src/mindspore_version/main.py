from gpt_mindspore import GPT2Model, GPT2Config

from model_expander_mindspore import expand_GPT

if __name__ == '__main__':
    target = GPT2Config(vocab_size=28996, num_hidden_layers=6, intermediate_size=120,
                        num_attention_heads=12)
    # 预训练bert
    org_model = GPT2Model(
        GPT2Config(vocab_size=28996, num_hidden_layers=3, intermediate_size=60,
                   num_attention_heads=6),
        is_training=True)
    new = expand_GPT(org_GPT=org_model, target_GPT_config=target, method="AKI")
    for params in new.get_parameters():
        print(params.name)
        print(params.value())
