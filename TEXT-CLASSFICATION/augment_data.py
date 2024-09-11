import random
import json



# Load the data from the JSONL file
input_file = 'all2.jsonl'
with open(input_file, 'r') as infile:
    data = [json.loads(line) for line in infile]

# Define all the augmentation methods (same as before)



synonyms = {
    "kitle": ["lezyon", "mass"],
    "fibroglandüler": ["fibroglandular", "fibroadenomatous"],
    "normal": ["standard", "olağan"],
    "patolojik": ["abnormal", "bozulmuş"],
    "yoğun": ["kalın", "dolu"],
    "cilt": ["ten", "epidermis"],
    "bilateral":["ikiyüzlü","ikitaraflı"],
    "gözlemlenmiştir":["izlendi" , "görüldü"],
    "izlendi":["görüldü" ,"gözlemlendi"],
    "görüldü":["izlendi" ,"gözlemlendi"],
    "görülmemiştir":["saptanmamıştır","izlenmemektedir"]
}



# Define all the augmentation methods
def character_swap(text, swap_prob=0.1):
    text_chars = list(text)
    for i in range(len(text_chars) - 1):
        if random.random() < swap_prob:
            text_chars[i], text_chars[i + 1] = text_chars[i + 1], text_chars[i]
    return ''.join(text_chars)


def character_replacement(text, replace_prob=0.1):
    text_chars = list(text)
    for i in range(len(text_chars)):
        if random.random() < replace_prob:
            text_chars[i] = random.choice('abcdefghijklmnoprstuvyzABCDEFGHIJKLMNOPRSTUVYZ')
    return ''.join(text_chars)



def character_deletion(text, delete_prob=0.1):
    text_chars = list(text)
    new_text_chars = [ch for ch in text_chars if random.random() > delete_prob]
    return ''.join(new_text_chars)



def character_insertion(text, insert_prob=0.1):
    text_chars = list(text)
    new_text_chars = []
    for ch in text_chars:
        new_text_chars.append(ch)
        if random.random() < insert_prob:
            new_text_chars.append(random.choice('abcdefghijklmnoprstuvyzABCDEFGHIJKLMNOPRSTUVYZ'))
    return ''.join(new_text_chars)



def combined_character_augmentation(text, noise_level=0.5):
    text = character_swap(text, noise_level)
    text = character_replacement(text, noise_level)
    text = character_deletion(text, noise_level)
    text = character_insertion(text, noise_level)
    return text



def synonym_replacement(text):
    words = text.split()
    new_words = []
    for word in words:
        if word in synonyms:
            new_words.append(random.choice(synonyms[word]))
        else:
            new_words.append(word)
    return ' '.join(new_words)




def augment_text(text, synonyms, method='combined', noise_level=0.1):
    if method == 'synonym':
        return synonym_replacement(text)
    elif method == 'swap':
        return character_swap(text, noise_level)
    elif method == 'replace':
        return character_replacement(text, noise_level)
    elif method == 'delete':
        return character_deletion(text, noise_level)
    elif method == 'insert':
        return character_insertion(text, noise_level)
    else:
        return combined_character_augmentation(text, noise_level)


def generate_augmented_samples(template, num_samples, method, synonyms, noise_level=0.04):
    augmented_samples = []
    for _ in range(num_samples):
        augmented_text = augment_text(template["text"], synonyms, method, noise_level)
        augmented_record = {
            "id": template['id'],
            "text": augmented_text,
            "label": template['label'],
            "Comments": template['Comments']
        }
        augmented_samples.append(augmented_record)
    return augmented_samples

# Apply augmentation to each entry in the data
augmentation_methods = ['swap', 'replace', 'delete', 'insert', 'combined', 'synonym']
samples_per_method = 2

augmented_samples = []

for entry in data:
    for method in augmentation_methods:
        augmented_samples.extend(generate_augmented_samples(entry, samples_per_method, method, synonyms))

# Save the augmented data back to a new JSONL file
augmented_jsonl = r"C:\Users\AKAY\Desktop\json_folder\all5.jsonl"
with open(augmented_jsonl, 'w') as outfile:
    for record in augmented_samples:
        json.dump(record, outfile)
        outfile.write('\n')



# Dictionary to map incorrect characters to correct Turkish characters
character_map = {
    "ÅŸ": "ş",
    "Åž": "Ş",
    "Ä±": "ı",
    "Ä°": "İ",
    "ÄŸ": "ğ",
    "Ğ": "Ğ",
    "Ã¼": "ü",
    "Ãœ": "Ü",
    "Ã¶": "ö",
    "Ã–": "Ö",
    "Ã§": "ç",
    "Ã‡": "Ç",
    "Ã": "ğ",  
    "¼": "ü",  
    "½": "ı",  
    "¤": "ş",  
    "§": "ç",  
    "¨": "ö",  
    "¢": "İ",  
    "¯": "Ü",  
    "°": "ğ",  
    "Â": "",   # No Turkish equivalent, likely an encoding artifact
    "\xa0": "", # No Turkish equivalent, non-breaking space
    "Ÿ": "",   # No Turkish equivalent, likely an encoding artifact
    "¶": "",   # No Turkish equivalent, pilcrow sign
    "‡": "",   # No Turkish equivalent, double dagger
    "™": "",   # No Turkish equivalent, trademark symbol
    "Å": "",   # No Turkish equivalent
    "Ä": "",   # No Turkish equivalent
    "â": "",   # No Turkish equivalent
    "±": "",   # No Turkish equivalent
    "€": "",   # No Turkish equivalent, Euro sign
    "“": "",   # No Turkish equivalent, opening double quotation mark
    "–": "",   # No Turkish equivalent, en dash
    "œ": "",   # No Turkish equivalent, oe ligature
}




def fix_encoding(text):
    for incorrect_char, correct_char in character_map.items():
        text = text.replace(incorrect_char, correct_char)
    return text

# Load the data
input_file_path = "all5.jsonl"  # Path to your input file
output_file_path = "all5.jsonl"  # Path to your output file

corrected_data = []

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        record = json.loads(line.strip())
        record['text'] = fix_encoding(record['text'])
        corrected_data.append(record)

# Save the corrected data
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for record in corrected_data:
        json.dump(record, outfile, ensure_ascii=False)
        outfile.write('\n')




print(f"Generated {len(augmented_samples)} augmented samples.")



print(f"Corrected data saved to {output_file_path}")