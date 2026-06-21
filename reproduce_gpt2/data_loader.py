import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B, self.T = B, T
        enc = tiktoken.get_encoding('gpt2')
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, '..', 'input.txt')) as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')
        
        self.current_pos = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos: self.current_pos + B*T + 1]
        
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        
        self.current_pos += B*T
        if self.current_pos + (B*T + 1) > len(self.tokens):
            self.current_pos = 0
            
        return x, y