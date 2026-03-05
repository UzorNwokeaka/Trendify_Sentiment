# Data pre-processing script
import re
import unicodedata
from typing import Optional

# Optional dependencies (recommended)
try:
    import ftfy  # fixes mojibake like "JAMÃS" -> "JAMÁS"
except ImportError:
    ftfy = None

try:
    import emoji  # converts emojis to text tokens like ":thumbs_up:"
except ImportError:
    emoji = None


_MOJIBAKE_HINTS = ("Ã", "â", "ðŸ", "�")


def _maybe_fix_mojibake(text: str) -> str:
    """
    Fix common encoding corruption (mojibake) seen in multilingual text.
    Uses ftfy if available, otherwise applies a safe heuristic.
    """
    if not text:
        return text

    if ftfy is not None:
        return ftfy.fix_text(text)

    # Fallback heuristic if ftfy isn't installed:
    # Only try if we see typical mojibake markers.
    if any(h in text for h in _MOJIBAKE_HINTS):
        try:
            # Common recovery path for UTF-8 bytes mis-decoded as latin-1
            return text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return text

    return text


def _normalize_unicode(text: str) -> str:
    """
    Normalize unicode to reduce weird variants (e.g., full-width chars).
    NFKC is a good general-purpose normalization for user-generated text.
    """
    return unicodedata.normalize("NFKC", text)


def _handle_emojis(text: str, mode: str = "demojize") -> str:
    """
    mode:
      - "demojize": 😀 -> :grinning_face:
      - "remove": 😀 -> ""
      - "keep": no change
    """
    if mode == "keep":
        return text

    if emoji is None:
        # If emoji lib isn't installed, fall back to removing non-text symbols only if requested
        if mode == "remove":
            return re.sub(r"[\U00010000-\U0010ffff]", "", text)
        return text

    if mode == "remove":
        # Remove emojis (and emoji-like symbols)
        return emoji.replace_emoji(text, replace="")

    # demojize: keep sentiment signal in text form
    return emoji.demojize(text, delimiters=(" ", " "))


def clean_text(
    text: Optional[str],
    *,
    lowercase: bool = True,
    fix_encoding: bool = True,
    emoji_mode: str = "demojize",
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_html: bool = True,
    keep_punctuation: bool = True,
) -> str:
    """
    Cleans and standardizes multilingual review text.
    Designed to be conservative (preserve meaning) while removing noise.
    """
    if text is None:
        return ""

    # Ensure string
    text = str(text)

    # Repair broken encoding + normalize unicode
    if fix_encoding:
        text = _maybe_fix_mojibake(text)
    text = _normalize_unicode(text)

    # Remove HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs / emails
    if remove_urls:
        text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
    if remove_emails:
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # Standardize apostrophes/quotes/dashes commonly seen in reviews
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    # Handle emojis
    text = _handle_emojis(text, mode=emoji_mode)

    # Remove control characters (newlines, tabs etc.) and normalize whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Optionally remove "special characters" while keeping multilingual letters
    # We keep letters/numbers/space plus selected punctuation.
    if keep_punctuation:
        # Keep basic punctuation that helps sentiment and structure
        text = re.sub(r"[^\w\s\.\,\!\?\:\;\-\(\)\'\"]+", " ", text, flags=re.UNICODE)
    else:
        # Keep only letters, numbers, and spaces
        text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)

    # Collapse repeated punctuation: "!!!" -> "!"
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)

    # Normalize repeated spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    if lowercase:
        # Lowercasing is generally OK for sentiment classification;
        # for transformers we may not need it, but it's fine for a clean standardized column.
        text = text.lower()

    return text
