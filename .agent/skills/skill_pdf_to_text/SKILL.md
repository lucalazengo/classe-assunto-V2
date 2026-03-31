---
name: skill_pdf_to_text
description: PDF text extraction and normalization for Brazilian legal petitions. Extracts clean text from scanned and native PDFs using pdfplumber or PyMuPDF.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# Skill: PDF to Text Extraction

> Extract and normalize text from legal petition PDFs in `modelos petições/`.
> Supports both native-digital PDFs and OCR-scanned documents.

---

## ⚠️ How to Use This Skill

This skill provides **extraction patterns and library recommendations** for converting PDF petitions into clean, analyzable text. The extracted text serves two purposes:
1. **Validate Nível 1 heuristics**: Test regex patterns against real petition headers
2. **Build test fixtures**: Create ground-truth text samples for unit tests

---

## 1. Library Selection

### Decision Tree

```
What type of PDF?
│
├── Native digital (text-selectable)
│   ├── pdfplumber (best for structured extraction, tables)
│   └── PyMuPDF/fitz (fastest, good text quality)
│
├── Scanned/OCR (image-based)
│   ├── PyMuPDF + OCR (built-in OCR support)
│   └── pytesseract + pdf2image (full OCR pipeline)
│
└── Mixed (some pages digital, some scanned)
    └── PyMuPDF with fallback to OCR
```

### Comparison

| Feature | pdfplumber | PyMuPDF (fitz) | pytesseract |
|---------|------------|----------------|-------------|
| **Speed** | Medium | Fast | Slow |
| **Text quality** | Excellent | Good | Variable |
| **Table extraction** | Built-in | Manual | No |
| **OCR support** | No | Optional | Yes |
| **Install size** | Small | Medium | Large (needs Tesseract) |

### Recommendation for This Project

**Use `pdfplumber`** as primary (petitions are mostly native-digital).
**Fallback to `PyMuPDF`** if pdfplumber fails on a specific PDF.

---

## 2. Extraction Pipeline

### Basic Extraction

```python
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    full_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    
    return '\n'.join(full_text)
```

### Extraction with Quality Check

```python
def extract_with_quality(pdf_path: str) -> dict:
    """
    Extract text and assess quality metrics.
    
    Returns:
        dict with keys: text, num_pages, char_count, 
        avg_chars_per_page, quality_score
    """
    pages_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    
    full_text = '\n'.join(pages_text)
    char_count = len(full_text)
    avg_chars = char_count / max(num_pages, 1)
    
    # Quality heuristic: good PDFs have > 200 chars/page avg
    quality = "good" if avg_chars > 200 else "poor" if avg_chars > 50 else "empty"
    
    return {
        'text': full_text,
        'num_pages': num_pages,
        'char_count': char_count,
        'avg_chars_per_page': round(avg_chars),
        'quality': quality,
    }
```

---

## 3. Batch Processing

```python
from pathlib import Path

def process_petition_folder(folder_path: str) -> list[dict]:
    """
    Process all PDFs in the modelos petições folder.
    
    Returns list of dicts with filename, extracted text, and quality metrics.
    """
    folder = Path(folder_path)
    results = []
    
    for pdf_file in sorted(folder.glob("*.pdf")):
        try:
            result = extract_with_quality(str(pdf_file))
            result['filename'] = pdf_file.name
            results.append(result)
            print(f"✓ {pdf_file.name}: {result['num_pages']} pages, "
                  f"{result['char_count']} chars ({result['quality']})")
        except Exception as e:
            results.append({
                'filename': pdf_file.name,
                'text': '',
                'error': str(e),
                'quality': 'error',
            })
            print(f"✗ {pdf_file.name}: {e}")
    
    return results
```

---

## 4. Header Extraction (For Nível 1 Validation)

```python
def extract_header(text: str, max_chars: int = 500) -> str:
    """
    Extract the header portion of a petition.
    The class/action type is typically declared in the first 500 chars.
    """
    return text[:max_chars].strip()

def extract_headers_from_folder(folder_path: str) -> dict[str, str]:
    """
    Extract headers from all petitions for Nível 1 pattern testing.
    
    Returns dict mapping filename to header text.
    """
    results = process_petition_folder(folder_path)
    headers = {}
    
    for r in results:
        if r.get('text'):
            headers[r['filename']] = extract_header(r['text'])
    
    return headers
```

---

## 5. Output Format

### Save Extracted Text

```python
import json

def save_extracted_texts(results: list[dict], output_path: str):
    """Save extraction results as JSON for downstream use."""
    # Save full results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary_path = output_path.replace('.json', '_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# PDF Extraction Summary\n\n")
        f.write(f"| File | Pages | Chars | Quality |\n")
        f.write(f"|------|-------|-------|---------|\n")
        for r in results:
            f.write(f"| {r.get('filename', 'N/A')} "
                    f"| {r.get('num_pages', 'N/A')} "
                    f"| {r.get('char_count', 'N/A')} "
                    f"| {r.get('quality', 'N/A')} |\n")
```

---

## 6. Dependencies

```
Required packages:
├── pdfplumber >= 0.10.0 (primary extractor)
├── PyMuPDF >= 1.23.0 (fallback, install as: pip install pymupdf)
└── Optional: pytesseract + pdf2image (for scanned PDFs only)
```

---

## 7. Integration Points

| Consumer | How They Use Extracted Text |
|----------|---------------------------|
| `nlp-engineer-specialist` | Test regex patterns against real petition headers |
| `eda-specialist` | Analyze text structure and quality of petition samples |
| `data-science-specialist` | Create test fixtures for model evaluation |
| `skill_legal_text_cleaner` | Input for cleaning pipeline validation |

---

> **Remember**: PDF extraction quality varies. Always check `quality` field and handle `poor`/`error` cases. For scanned PDFs, OCR may produce lower quality text that needs heavier cleaning.
