
import json 

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
input_file_path = "all5_corrected.jsonl"  # Path to your input file
output_file_path = "all5_corrected.jsonl"  # Path to your output file


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


print(f"Corrected data saved to {output_file_path}")