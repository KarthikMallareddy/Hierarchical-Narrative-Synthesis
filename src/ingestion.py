import pandas as pd
from io import StringIO
from pdfminer.high_level import extract_text

def parse_pdf(file):
    """
    Extract text from a PDF file using pdfminer.six.
    """
    try:
        text = extract_text(file)
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def parse_csv(file):
    """
    Load CSV into a pandas DataFrame and convert to text representation.
    """
    try:
        df = pd.read_csv(file)
        # Convert dataframe to a string representation for embedding
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

def parse_txt(file):
    """
    Read text from a plain text file.
    """
    try:
        # Streamlit UploadedFile object needs to be read and decoded
        return file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def ingest_file(uploaded_file):
    """
    Dispatcher for file ingestion based on extension.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return parse_pdf(uploaded_file)
    elif filename.endswith(".csv"):
        return parse_csv(uploaded_file)
    elif filename.endswith(".txt") or filename.endswith(".log"):
        return parse_txt(uploaded_file)
    else:
        return "Unsupported file format"


def process_uploaded_files(uploaded_files):
    """
    Process a list of Streamlit uploaded files into text segments.
    Each file is ingested and chunked into ~200-word segments.
    Returns a flat list of text segments ready for embedding.
    """
    all_segments = []
    for f in uploaded_files:
        raw_text = ingest_file(f)
        if raw_text and not raw_text.startswith("Error") and not raw_text.startswith("Unsupported"):
            # Chunk into segments
            words = raw_text.split()
            for i in range(0, len(words), 200):
                chunk = " ".join(words[i:i+200])
                if len(chunk.strip()) > 20:
                    all_segments.append(chunk)
    return all_segments
