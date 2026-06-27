import torch
from transformers import GPT2LMHeadModel
from torch.nn import functional as F

from model import GPT, GPTConfig
from data_loader import DataLoaderLite


def load_weight():
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    for k,v in sd_hf.items():
        print(k, v.shape)

def generate():
    max_len = 30
    num_return_seqs = 5
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"using device: {device}")
    
    model = GPT.from_pretrained('gpt2')
    # Random weight
    # model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    print("didn't crach yet~")
    
    # input prefix tokens
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a llm,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_seqs, 1) # (5, 8)
    x = tokens.to(device)
    
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

import math
learning_rate = 6e-4
min_lr = learning_rate * 0.1
warmup_iters = 10
lr_decay_iters = 50

# GPT-2/3训练细节
## learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

## BatchSize在初始时小，逐渐增大 (未实现)。因为一开始学习的东西差不多，丢任何数据loss都很大
## 不重复Sample数据，直到下一个epoch。最小化过拟合

def train():
    device = "cpu"
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(1337)
    print(f"using device: {device}")
    
    import time
    B, T = 16, 128
    dl = DataLoaderLite(B, T)
    # 优化1: GPU(A100)默认FP32(8+23 bit)，改为TF32(8+10 bit)
    # 问题: 实际吞吐的收益没有达到8X的预期，因为Memory Bound
    torch.set_float32_matmul_precision('high')
    
    # 优化5: 数字改为能被2的幂次整除的数, 会略微增加内存和计算量
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    # 优化点3: torch.compile 为什么比 eager 模式快
    # 1. 算子融合（Kernel Fusion）：eager模式下每个op是一次GPU kernel launch，有大量内存读写开销。
    # 编译后多个op融合成单个kernel，减少中间结果读写，典型如matmul + bias + activation 融合为一个kernel。
    # 2. 反向传播也编译：AOTAutograd让backward不再是逐op执行，同样得到kernel fusion优化。
    # 3. 减少Python开销：编译后跳过Python解释器逐行执行的开销。
    model = torch.compile(model)
    
    # optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    
    # weight_decay: 防止权重矩阵过拟合，又保护了bias和LayerNorm的正常功能
    # FusedAdamW: AdamW更新每个参数需要：读4个tensor（weight、grad、m、v），写3个tensor（weight、m、v）。
    # 对于一个有数千个参数的GPT模型，普通AdamW要启动数千个小kernel，每个都有launch开销和内存读写开销。
    # Fused版把这些全部打包成一个kernel，减少GPU kernel launch开销和内存往返次数。
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), 
                                           device_type=device)
    for step in range(lr_decay_iters):
        t0 = time.time()
        optimizer.zero_grad()
        x, y = dl.next_batch()
        x, y = x.to(device), y.to(device)
        # 优化点2: 使用低精度bf16(8+7)权重参数
        # 问题: 没有用float16(fp16 5+10), 因为FP16 (5+10) 的5位指数导致动态范围太小，最大值仅约65500。
        #   训练中梯度很容易溢出（overflow）或下溢（underflow），必须配合 torch.cuda.amp.GradScaler
        #   来缩放梯度——先放大梯度防溢出，反向传播后再缩回来再更新权重。
        #   BF16 (8+7) 的8位指数与FP32相同，动态范围和FP32一致，梯度不会轻易溢出/下溢
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # 部分权重被自动转为bf16(主要是mutmal)，像softmax等操作受精度影响较大，pytorch没有转换
            # import code; code.interact(local=locals())
            #   >>> logits.dtype
            #   torch.bfloat16
            #   >>> model.transformer.wte.weight.dtype
            #   torch.float32
        loss.backward()
        # clip_grad_norm_作用是梯度裁剪（Gradient Clipping），防止梯度爆炸
        # 具体机制
        #   1. 计算所有参数梯度的 L2范数（即所有梯度平方和的平方根）
        #   2. 如果范数 > max_norm=1.0，将所有梯度等比例缩放，使缩放后的范数恰好等于1.0
        #   3. 如果范数 ≤ 1.0，不做任何操作
        # 函数返回值 norm 是裁剪前的梯度范数，可以用来监控训练是否稳定。
        # Transformer训练中容易出现梯度爆炸——某些step的梯度突然变得极大，导致参数更新幅度过ss大，
        # 模型崩溃（loss飙升）。梯度裁剪给了一个安全上限，确保单次更新的步长不会超过阈值。
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        # Metrics
        torch.cuda.synchronize()    # wait for the GPU to finish work
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (B*T)/(t1-t0)
        print(f'step {step} | loss: {loss.item()} | lr: {lr:.4f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}')

if __name__ == "__main__":
    # load_weight()
    # generate()
    train()
