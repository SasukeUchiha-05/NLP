import pandas as pd

# Paths to your files
english_file_path = 'ELRC-EMEA.en-es.en'
spanish_file_path = 'ELRC-EMEA.en-es.es'  # Adjust if your filename is different

# Read the files line by line
with open(english_file_path, 'r', encoding='utf-8') as en_file:
    english_sentences = en_file.read().splitlines()

with open(spanish_file_path, 'r', encoding='utf-8') as es_file:
    spanish_sentences = es_file.read().splitlines()

# Check if both files have the same number of lines
assert len(english_sentences) == len(spanish_sentences), "Line count mismatch between files!"

# Create the DataFrame
df = pd.DataFrame({
    'english': english_sentences,
    'spanish': spanish_sentences
})

# Preview the DataFrame
print(df.head())

# Optional: Save to CSV
df.to_csv('bilingual_dataset.csv', index=False)
