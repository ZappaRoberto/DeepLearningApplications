from pathlib import Path
from tokenizers import SentencePieceBPETokenizer


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
    main()
