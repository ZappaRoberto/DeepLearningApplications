from pathlib import Path
from tokenizers import SentencePieceBPETokenizer, Tokenizer
import torch


class Tokenization:
    def __init__(self):
        self.tokenizer = Tokenizer.from_file("dante.tokenizer.json")

    def encode(self, string):
        out = self.tokenizer.encode(string)
        out = torch.tensor(out.ids).unsqueeze(dim=0)
        return out

    def decode(self, tensor):
        out = self.tokenizer.decode(tensor.tolist())
        return out

    def train(self, string):
        out = self.encode(string).unsqueeze(dim=0)
        pass

    def inference(self, string):
        pass

def main():
    paths = [str(x) for x in Path("./Dataset").glob("**/*.txt")]
    tokenizer = SentencePieceBPETokenizer().train(
        files=paths,
        vocab_size=10000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save("./dante.tokenizer.json")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("dante.tokenizer.json")
    inputs = tokenizer.encode('ciao')
    print(inputs.ids)
    print(tokenizer.decode(inputs.ids))
    # main()
