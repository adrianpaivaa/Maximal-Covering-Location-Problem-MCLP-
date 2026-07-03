import fitz
import os

pdf_path = r'c:\Users\adria\Desktop\TrabalhoFinal-PO\1-s2.0-S1056819023021395-main.pdf'
output_path = r'c:\Users\adria\Desktop\TrabalhoFinal-PO\article_text.txt'

doc = fitz.open(pdf_path)
text = ''
for i, page in enumerate(doc):
    text += f'\n--- PAGE {i+1} ---\n'
    text += page.get_text()

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Extracted {len(doc)} pages, {len(text)} characters")
