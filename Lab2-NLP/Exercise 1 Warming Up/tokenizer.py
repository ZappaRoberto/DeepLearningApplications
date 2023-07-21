from pathlib import Path
from tokenizers import SentencePieceBPETokenizer, Tokenizer
import torch
import torch.nn.functional as F


class Tokenization:
    def __init__(self):
        self.tokenizer = Tokenizer.from_file("dante.tokenizer.json")

    def encode(self, string):
        out = self.tokenizer.encode(string)
        out = torch.tensor(out.ids[:-1]).unsqueeze(dim=0)
        return out

    def decode(self, tensor):
        out = F.softmax(tensor, dim=-1)
        out = torch.argmax(out, dim=-1)
        out = self.tokenizer.decode(out.tolist())
        return out

    def generate(self, string):
        out = self.tokenizer.encode(string)
        out = torch.tensor(out.ids).unsqueeze(dim=0)
        return out


def main():
    paths = [str(x) for x in Path("./Dataset").glob("**/*.txt")]
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=5568,
        min_frequency=2,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save_model("./Tokenizer", "dante")


if __name__ == "__main__":
    # tokenizer = Tokenization()
    # inputs = tokenizer.encode('ciao')
    # print(inputs.shape)
    # print(tokenizer.decode(inputs))
    main()
