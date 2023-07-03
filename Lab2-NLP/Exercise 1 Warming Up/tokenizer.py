from pathlib import Path
from tokenizers import SentencePieceBPETokenizer, Tokenizer


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
    print("Hello World!")
    tokenizer = Tokenizer.from_file("dante.tokenizer.json")
    inputs = tokenizer.encode('ciao')
    print(inputs.ids)
    # main()
