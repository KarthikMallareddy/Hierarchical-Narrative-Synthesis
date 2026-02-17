
def clean_text(text):
    """
    Basic text cleaning utility.
    """
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
