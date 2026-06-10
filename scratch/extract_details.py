import pypdf

def extract_pages(pdf_path, page_indices, output_txt):
    reader = pypdf.PdfReader(pdf_path)
    with open(output_txt, "w") as f:
        for idx in page_indices:
            if idx < len(reader.pages):
                f.write(f"--- PAGE {idx + 1} ---\n")
                f.write(reader.pages[idx].extract_text() or "")
                f.write("\n\n")

# Joachim Klement's model page 4-6
extract_pages("docs/strs_1031724.pdf", [3, 4, 5], "scratch/klement_model.txt")
print("Extracted Joachim Klement model pages.")

# Sylla & Magel's model page 3-7
extract_pages("docs/Predicting_the_Winner_of_Games_in_World_Cup_Soccer(1).pdf", [2, 3, 4, 5, 6], "scratch/sylla_model.txt")
print("Extracted Sylla & Magel model pages.")
