from transformers import GPT2Tokenizer


def main(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    encoded_input = tokenizer(text, return_tensors="pt")

    # Compare the length of input text with the encoded sequence length
    input_length = len(text.split())
    encoded_length = encoded_input["input_ids"].shape[1]

    print(f"Original text length: {input_length}")
    print(f"Encoded sequence length: {encoded_length}")

    decoded_text = tokenizer.decode(encoded_input["input_ids"][0])
    print(f"Decoded text: {decoded_text}")


if __name__ == "__main__":
    # Text to encode
    text = "This is an example sentence to encode into tokens."
    main(text)
