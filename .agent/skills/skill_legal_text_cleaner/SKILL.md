---
name: skill_legal_text_cleaner
description: Legal text preprocessing and cleaning pipeline for Brazilian court documents. Removes OCR artifacts, legal stopwords, mojibakes, and normalizes text for ML pipelines.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# Skill: Legal Text Cleaner

> Preprocessing pipeline for Brazilian legal text (petitions, rulings, court documents).
> Produces clean, normalized text ready for vectorization or model inference.

---

## ⚠️ How to Use This Skill

This skill defines **preprocessing rules and patterns** for legal text. Apply them in sequence as a pipeline. Each step is independent and can be toggled based on the downstream task.

---

## 1. Text Markers & Artifacts Removal

### Document Boundary Markers

```python
# Remove PJe document markers
import re

def remove_markers(text: str) -> str:
    """Remove >>>>>inicio<<<<< and similar boundary markers."""
    text = re.sub(r'>{3,}\w+<{3,}', '', text)
    text = re.sub(r'<{3,}\w+>{3,}', '', text)
    return text.strip()
```

### HTML/XML Residuals

```python
def remove_html(text: str) -> str:
    """Remove HTML tags left from document conversion."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)  # &nbsp; &amp; etc.
    return text
```

---

## 2. Encoding Fixes (Mojibakes)

### Strategy

```
OCR and legacy systems produce encoding errors:
├── "Ã§Ã£o" → "ção" (UTF-8 read as Latin-1)
├── "Ã©" → "é"
├── "â€" → "–" (smart quotes/dashes)
└── Use `ftfy` library as first-pass fix
```

### Implementation

```python
import ftfy

def fix_encoding(text: str) -> str:
    """Fix mojibake encoding issues from OCR/legacy systems."""
    return ftfy.fix_text(text)
```

---

## 3. Legal Stopwords

### Domain-Specific Stop Phrases

```python
LEGAL_STOP_PHRASES = [
    # Honorifics
    "excelentíssimo senhor", "excelentíssima senhora",
    "meritíssimo juiz", "meritíssima juíza",
    "douto juízo", "colenda câmara",
    # Procedural formulas
    "vem respeitosamente", "vem perante",
    "nos autos do processo", "nos termos da lei",
    "conforme disposto no artigo", "à presença de vossa excelência",
    "requer a vossa excelência", "pede deferimento",
    # Institutional headers
    "poder judiciário", "tribunal de justiça",
    "estado de goiás", "comarca de",
    "procuradoria geral", "defensoria pública",
    "ministério público",
]

def remove_legal_stopwords(text: str) -> str:
    """Remove common legal formulaic expressions."""
    text_lower = text.lower()
    for phrase in LEGAL_STOP_PHRASES:
        text_lower = text_lower.replace(phrase, ' ')
    return text_lower
```

> [!WARNING]
> **Do NOT remove legal stopwords for Nível 1 (Heuristic)**. The heuristic relies on exact patterns like "AÇÃO DE COBRANÇA" which include these terms. Only apply stopword removal for statistical/ML models (Nível 2 and 3).

---

## 4. Whitespace & Formatting Normalization

```python
def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, newlines, and tabs into single spaces."""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()
```

---

## 5. Number & Special Character Handling

```python
def normalize_numbers(text: str) -> str:
    """Replace specific number patterns with tokens."""
    # CPF/CNPJ
    text = re.sub(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}', ' DOC_CPF ', text)
    text = re.sub(r'\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}', ' DOC_CNPJ ', text)
    # Process numbers
    text = re.sub(r'\d{7}-?\d{2}\.?\d{4}\.?\d\.?\d{2}\.?\d{4}', ' NUM_PROCESSO ', text)
    # Currency
    text = re.sub(r'R\$\s?[\d.,]+', ' VALOR_MONETARIO ', text)
    # Generic long numbers
    text = re.sub(r'\d{5,}', ' NUM_LONGO ', text)
    return text
```

---

## 6. Full Pipeline

```python
def clean_legal_text(text: str, for_heuristic: bool = False) -> str:
    """
    Full cleaning pipeline for legal text.
    
    Args:
        text: Raw petition text
        for_heuristic: If True, skip stopword removal (Nível 1 needs them)
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Fix encoding
    text = fix_encoding(text)
    
    # Step 2: Remove markers
    text = remove_markers(text)
    
    # Step 3: Remove HTML
    text = remove_html(text)
    
    # Step 4: Normalize whitespace
    text = normalize_whitespace(text)
    
    if not for_heuristic:
        # Step 5: Remove legal stopwords (only for ML models)
        text = remove_legal_stopwords(text)
        
        # Step 6: Normalize numbers
        text = normalize_numbers(text)
    
    return text.strip()
```

---

## 7. Decision Guide

| Downstream Task | Apply Stopwords? | Apply Number Normalization? | Lower Case? |
|-----------------|-------------------|-----------------------------|-------------|
| Nível 1 (Heuristic) | ❌ No | ❌ No | ❌ No (case-insensitive regex) |
| Nível 2 (TF-IDF + LightGBM) | ✅ Yes | ✅ Yes | ✅ Yes |
| Nível 3 (Transformer) | ❌ No | ❌ No | Depends on tokenizer |
| EDA / Analysis | ❌ No | ❌ No | ❌ No (preserve original) |

---

## 8. Dependencies

```
Required packages:
├── ftfy >= 6.0 (encoding fixes)
├── re (standard library)
└── Optional: unidecode (accent removal for specific cases)
```

---

> **Remember**: Cleaning is context-dependent. Always ask "what will consume this text?" before deciding which steps to apply.
