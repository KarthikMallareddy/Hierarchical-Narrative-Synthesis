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
