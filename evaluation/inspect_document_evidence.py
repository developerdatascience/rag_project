
# evaluation/inspect_document_evidence.py

import re

from src.loader import load_document


PDF_PATH = "data/documents/understanding-the-dom.pdf"

SEARCH_TERMS = {
    "javascript_alert": [
        r"\balert\s*\(",
        r"\bwindow\.alert\b",
        r"alert message",
        r"addEventListener",
    ],
    "javascript_api_fetch": [
        r"\bfetch\s*\(",
        r"\bXMLHttpRequest\b",
        r"\bREST API\b",
        r"\bAPI request\b",
        r"\bapplication programming interface\b",
    ],
    "india_capital": [
        r"\bcapital of India\b",
        r"\bNew Delhi\b",
    ],
}


def find_matches(text: str, pattern: str):
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for match in regex.finditer(text):
        start = max(0, match.start() - 180)
        end = min(len(text), match.end() + 240)
        yield match.group(), text[start:end].replace("\n", " ")


def main():
    pages = load_document(PDF_PATH)
    total_matches = 0

    for case_id, patterns in SEARCH_TERMS.items():
        print("\n" + "=" * 80)
        print(f"CASE: {case_id}")
        print("=" * 80)

        case_matches = 0

        for document in pages:
            page = document.metadata.get("page")
            page_number = (
                int(page) + 1 if isinstance(page, int) else "unknown"
            )

            for pattern in patterns:
                for matched_text, snippet in find_matches(
                    document.page_content, pattern
                ):
                    case_matches += 1
                    total_matches += 1

                    print(f"\nPage: {page_number}")
                    print(f"Matched text: {matched_text}")
                    print(f"Context: ...{snippet}...")

        if case_matches == 0:
            print("No matching terms found in extracted PDF text.")

    print(f"\nTotal matches: {total_matches}")
    print(
        "\nNote: no matches does not prove a topic is absent. "
        "The document may use different wording or contain scanned pages."
    )


if __name__ == "__main__":
    main()
