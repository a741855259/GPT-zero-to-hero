import torch
from transformers import GPT2LMHeadModel
from torch.nn import functional as F

from model import GPT


def load_weight():
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    for k,v in sd_hf.items():
        print(k, v.shape)

def generate():
    max_len = 30
    num_return_seqs = 5
    
    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to('cuda')
    print("didn't crach yet~")
    
    # input prefix tokens
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a llm,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_seqs, 1) # (5, 8)
    x = tokens.to('cuda')
    
    # generate B = 5, T = 8
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_len:
        with torch.no_grad():
            logits = model(x)   # (b, t, vocab_size)
            # 取每个序列中最后一个新生成的token
            logits = logits[:, -1, :]   # (b, vocab_size)
            # 计算概率
            probs = F.softmax(logits, dim=-1)
            # topk采样
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # 从k个概率中取一个token
            ix = torch.multinomial(topk_probs, 1)   # (b, 1)
            # 汇聚所有所取token在词表中的索引
            xcol = torch.gather(topk_indices, -1, ix)    # (b, 1)
            # 添加到原始序列最后
            x = torch.cat((x, xcol), dim=1)
    
    # 打印结果
    for i in range(num_return_seqs):
        tokens = x[i, :max_len].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    # load_weight()
    generate()

