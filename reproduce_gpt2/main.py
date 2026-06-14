from transformers import GPT2LMHeadModel

def main():
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    for k,v in sd_hf.items():
        print(k, v.shape)

if __name__ == "__main__":
    main()

