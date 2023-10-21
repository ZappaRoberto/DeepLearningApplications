from textwrap import wrap

def main(input_file):
    with open(input_file, "r", encoding='utf-8') as file:
        text = file.read()
    cleaned_text = text.replace(">>", '"')
    cleaned_text = cleaned_text.replace("<<", '"')
    cleaned_text = cleaned_text.replace("  ", " ")
    cleaned_text = cleaned_text.replace("\n ", "\n")
    cleaned_text = cleaned_text.replace("\\'", "'")

    lines = cleaned_text.split("\n")
    lines = [line for line in lines if line.strip() != ""]
    cleaned_text = "\n".join(lines)
    output_file = 'dataset.txt'
    with open(output_file, "w") as file:
        file.write(cleaned_text)


if __name__ == '__main__':
    main('Divina Commedia.txt')