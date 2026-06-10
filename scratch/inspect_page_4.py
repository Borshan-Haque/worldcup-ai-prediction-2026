import pypdf

reader = pypdf.PdfReader("docs/The-World-Cup-and-Economics_-World-Cup-2026_-Predictions-Probabilities-and-Paths-to-Victory.pdf")
print("Page 4 Text:")
print(reader.pages[3].extract_text())
print("\nPage 5 Text:")
print(reader.pages[4].extract_text())
