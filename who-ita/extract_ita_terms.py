"""
Extract ITA terms from WHO-ITA_2022.pdf

Reads the WHO International Standard Terminologies on Ayurveda PDF and
extracts Term ID, English term, and Sanskrit term in IAST into a CSV file.

The PDF has a consistent 5-column tabular layout on content pages 19-491,
with horizontal ruling lines separating entries. We use pdfplumber's table
extraction with explicit vertical column boundaries and line-based horizontal
row detection for accurate multi-line entry grouping.

Usage:
    python extract_ita_terms.py
"""

import csv
import re
import pdfplumber

PDF_PATH = "WHO-ITA_2022.pdf"
OUTPUT_CSV = "ita_terms.csv"

# 0-indexed page range: page 19 through page 491
FIRST_PAGE_IDX = 18
LAST_PAGE_IDX = 490

# Explicit vertical column boundaries (from PDF ruling lines)
# Columns: Term ID | English term | Description | Sanskrit IAST | Sanskrit Devanagari
VERTICAL_LINES = [57.6, 121.0, 232.6, 523.4, 653.0, 784.3]

# Table extraction settings
TABLE_SETTINGS = {
    "vertical_strategy": "explicit",
    "explicit_vertical_lines": VERTICAL_LINES,
    "horizontal_strategy": "lines",
    "snap_y_tolerance": 5,
}

# Column indices in the extracted table rows
COL_TERM_ID = 0
COL_ENGLISH = 1
COL_IAST = 3

# Pattern for ITA term IDs: ITA-x.x.x (with variable depth)
ITA_ID_PATTERN = re.compile(r"^ITA-\d+\.\d+")


def cell_text(cell):
    """Normalize a table cell: collapse newlines into spaces, strip whitespace."""
    if not cell:
        return ""
    return " ".join(cell.split()).strip()


def _words_for_ita_id(page, target_id):
    """Fallback: extract English and IAST for an ITA ID using word positions."""
    words = page.extract_words(keep_blank_chars=True)
    words.sort(key=lambda w: (w["top"], w["x0"]))
    words = [w for w in words if w["top"] < 550 and w["x0"] < 780]

    # Find the row(s) for this ITA ID
    id_word = None
    for w in words:
        if w["text"].strip() == target_id and w["x0"] < VERTICAL_LINES[1]:
            id_word = w
            break
    if not id_word:
        return "", ""

    # Collect words at similar or slightly later top positions (same entry)
    id_top = id_word["top"]
    entry_words = [w for w in words if id_top - 2 <= w["top"] < id_top + 30]

    english_parts = []
    iast_parts = []
    for w in sorted(entry_words, key=lambda w: (w["top"], w["x0"])):
        x0 = w["x0"]
        if VERTICAL_LINES[1] <= x0 < VERTICAL_LINES[2]:
            english_parts.append(w["text"])
        elif VERTICAL_LINES[3] <= x0 < VERTICAL_LINES[4]:
            iast_parts.append(w["text"])

    return " ".join(english_parts).strip(), " ".join(iast_parts).strip()


def extract_terms(pdf_path):
    """Extract ITA terms from the PDF using table extraction."""
    entries = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(FIRST_PAGE_IDX, LAST_PAGE_IDX + 1):
            page = pdf.pages[page_idx]
            tables = page.extract_tables(TABLE_SETTINGS)

            for table in tables:
                for row in table:
                    if not row or len(row) < 4:
                        continue

                    term_id = cell_text(row[COL_TERM_ID])
                    english = cell_text(row[COL_ENGLISH])
                    iast = cell_text(row[COL_IAST])

                    # Skip header rows
                    if term_id == "Term ID":
                        continue

                    # Only keep rows with a valid ITA ID
                    if not ITA_ID_PATTERN.match(term_id):
                        continue

                    # Fallback: if table extraction missed content (e.g., entry
                    # at page boundary below last horizontal line), use words
                    if not english and not iast:
                        english, iast = _words_for_ita_id(page, term_id)

                    entries.append({
                        "ITA_ID": term_id,
                        "English_Term": english,
                        "Sanskrit_IAST": iast,
                    })

    return entries


def clean_entries(entries):
    """Clean up extracted entries and merge duplicates from multi-page spans."""
    # Merge consecutive entries with the same ITA ID (multi-page descriptions)
    merged = []
    for entry in entries:
        if merged and merged[-1]["ITA_ID"] == entry["ITA_ID"]:
            # Append English and IAST from the continuation
            if entry["English_Term"]:
                prev = merged[-1]["English_Term"]
                merged[-1]["English_Term"] = (prev + " " + entry["English_Term"]).strip()
            if entry["Sanskrit_IAST"]:
                prev = merged[-1]["Sanskrit_IAST"]
                merged[-1]["Sanskrit_IAST"] = (prev + " " + entry["Sanskrit_IAST"]).strip()
        else:
            merged.append(entry)

    # Clean up text
    for entry in merged:
        # Rejoin hyphenated words split across lines (e.g., "pharma- cology" -> "pharmacology")
        entry["English_Term"] = re.sub(r"(\w)- (\w)", r"\1\2", " ".join(entry["English_Term"].split()))
        iast = re.sub(r"(\w)- (\w)", r"\1\2", " ".join(entry["Sanskrit_IAST"].split()))
        iast = iast.rstrip(";").rstrip(".").strip()
        entry["Sanskrit_IAST"] = iast

    return merged


def write_csv(entries, output_path):
    """Write entries to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ITA_ID", "English_Term", "Sanskrit_IAST"])
        writer.writeheader()
        writer.writerows(entries)


def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print(f"Extracting ITA terms from {PDF_PATH}...")
    entries = extract_terms(PDF_PATH)
    entries = clean_entries(entries)
    write_csv(entries, OUTPUT_CSV)
    print(f"Extracted {len(entries)} entries to {OUTPUT_CSV}")

    # Quick summary
    if entries:
        print(f"\nFirst 5 entries:")
        for e in entries[:5]:
            print(f"  {e['ITA_ID']:20s}  {e['English_Term'][:40]:40s}  {e['Sanskrit_IAST']}")
        print(f"\nLast 5 entries:")
        for e in entries[-5:]:
            print(f"  {e['ITA_ID']:20s}  {e['English_Term'][:40]:40s}  {e['Sanskrit_IAST']}")


if __name__ == "__main__":
    main()
