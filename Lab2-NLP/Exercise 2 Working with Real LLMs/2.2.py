from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def generate_text(prompt, model_name="gpt2", max_length=50, num_return_sequences=1, do_sample=True, top_k=50,
                  temperature=0.7):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    torch.manual_seed(42)  # Set the seed for reproducibility

    generated_text = model.generate(
        input_ids=tokenizer.encode(prompt, return_tensors="pt"),
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return generated_text


if __name__ == '__main__':
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print(generated_text)
