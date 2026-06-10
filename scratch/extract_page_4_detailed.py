import pypdf

reader = pypdf.PdfReader("docs/The-World-Cup-and-Economics_-World-Cup-2026_-Predictions-Probabilities-and-Paths-to-Victory.pdf")
page = reader.pages[3]

def visitor_body(text, cm, tm, font_dict, font_size):
    if text.strip():
        print(f"Text: '{text.strip()}' at position cm={cm}, tm={tm}, size={font_size}")

page.extract_text(visitor_text=visitor_body)
