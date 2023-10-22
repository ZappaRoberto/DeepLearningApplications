from pathlib import Path
from tokenizers import SentencePieceBPETokenizer, Tokenizer
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn.functional as F


class Tokenization:
    def __init__(self, max_token_len=1024):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file="Tokenizer/tokenizer.json",
                                                 unk_token="<unk>",
                                                 eos_token="</s>",
                                                 bos_token="<s>",
                                                 pad_token="<pad>",
                                                 mask_token="<mask>")
        self.max_token_len = max_token_len

    def encode(self, string):
        tokens = self.tokenizer.__call__(string,
                                         add_special_tokens=True,
                                         return_tensors='pt',
                                         truncation=True,
                                         padding='max_length',
                                         max_length=self.max_token_len,
                                         return_attention_mask=True)
        return tokens.input_ids.flatten(), tokens.attention_mask.flatten()

    def decode(self, tensor):
        return self.tokenizer.decode(tensor)

    def dataset_preparation(self, dataset):
        tokens = self.tokenizer.__call__(dataset,
                                         add_special_tokens=True,
                                         return_tensors='pt',
                                         truncation=False,
                                         return_attention_mask=False)
        return tokens.input_ids.flatten()


def main():
    paths = [str(x) for x in Path("./Dataset").glob("**/*.txt")]
    tokenizer = SentencePieceBPETokenizer(add_prefix_space=True)
    tokenizer.train(
        files=paths,
        vocab_size=5549,
        min_frequency=2,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save("tokenizer.json", pretty=True)


if __name__ == "__main__":
    tokenizer = Tokenization()
    print(tokenizer.dataset_preparation('Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura'))
    #tokens, attention = tokenizer.encode('Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura')
    #print(tokens, attention)
    #print(tokenizer.decode(tokens))
    #main()

