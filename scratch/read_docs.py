import pypdf
import os

def summarize_pdf(pdf_path):
    print(f"\n========================================\nSUMMARIZING: {os.path.basename(pdf_path)}\n========================================")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    
    # Print first page text
    print("--- Page 1 (Abstract/Introduction) ---")
    print(reader.pages[0].extract_text()[:1500])
    
    # Search for mathematical terms
    terms = ["bivariate", "poisson", "dixon", "elo", "logistic", "regression", "xg", "brier", "calibration", "ensemble"]
    found_terms = {}
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        text_lower = text.lower()
        for term in terms:
            if term in text_lower:
                if term not in found_terms:
                    found_terms[term] = []
                found_terms[term].append(i + 1)
                
    print("\n--- Key Terms Found On Pages ---")
    for term, pages in found_terms.items():
        print(f"- '{term}': Pages {pages[:8]} (total {len(pages)} occurrences)")
        
    # Search and print pages containing formulas/methods
    print("\n--- Snippets from pages with methodology or estimation ---")
    methodology_printed = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        text_lower = text.lower()
        if "model" in text_lower or "estimation" in text_lower or "regression" in text_lower or "poisson" in text_lower:
            # Check if there are equations or tables
            if any(char in text for char in ["=", "+", "-", "*", "/", "lambda", "mu", "theta", "beta"]):
                print(f"\n[Page {i + 1} Snippet]:")
                print(text[:800])
                print("-" * 40)
                methodology_printed += 1
                if methodology_printed >= 2:
                    break

summarize_pdf("docs/Predicting_the_Winner_of_Games_in_World_Cup_Soccer(1).pdf")
summarize_pdf("docs/strs_1031724.pdf")
