import re
from datetime import date
from typing import Optional

from dateutil.parser import ParserError
from dateutil.parser import parse as dateutil_parse

start_pattern = r"(?:^|\s)"
end_pattern = r"(?:[;,.]?\s|$)"


NUMERICAL_PATTERNS = [
    r"(?:^|\s)(\d{1,2}/\d{1,2}/\d{4})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{1,2}/\d{1,2}/\d{2})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{4}/\d{1,2}/\d{1,2})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{1,2}-\d{1,2}-\d{4})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{1,2}-\d{1,2}-\d{2})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{4}-\d{1,2}-\d{1,2})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{4}\.\d{1,2}\.\d{1,2})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{1,2}\.\d{1,2}\.\d{4})(?:[;,.]?\s|$)",
    r"(?:^|\s)(\d{1,2}\.\d{1,2}\.\d{2})(?:[;,.]?\s|$)",
]

TEXT_PATTERNS = [
    # Month name (full and abbreviated) followed by day and year
    r"(?:^|\s)((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})(?:[;,.]?\s|$)",  # Added ordinal suffixes
    # Day followed by month name (full and abbreviated) and year
    r"(?:^|\s)(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})(?:[;,.]?\s|$)",  # Added ordinal suffixes
    # Year followed by month name (full and abbreviated) and day
    r"(?:^|\s)(\d{4}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2})(?:[;,.]?\s|$)",
]


DEFAULT_PATTERNS = NUMERICAL_PATTERNS + TEXT_PATTERNS


def register_date_pattern(pattern: str):
    """
    Registers a new date pattern to be used by the date extraction
    functions
    """
    if pattern not in DEFAULT_PATTERNS:
        DEFAULT_PATTERNS.append(pattern)


def get_date_pattern_regexes():
    return [re.compile(x, re.IGNORECASE) for x in DEFAULT_PATTERNS]


def get_date_matches_for_text(
    text: str, unique: bool = False, patterns: Optional[list[re.Pattern]] = None
) -> list[re.Match]:
    results = []

    patterns = patterns or get_date_pattern_regexes()

    for regex in patterns:
        for match in regex.finditer(text):
            results.append(match)

    if unique:
        return list(set(results))

    return results


def get_date_strings_for_text(
    text: str, unique: bool = False, patterns: Optional[list[re.Pattern]] = None
) -> list[str]:
    results = []

    patterns = patterns or get_date_pattern_regexes()

    for regex in patterns:
        for match_text in regex.findall(text):
            results.append(match_text.strip())

    if unique:
        return list(set(results))

    return results


def get_dates_for_text(text: str, unique: bool = False, patterns: Optional[list[re.Pattern]] = None) -> list[date]:
    results = []

    patterns = patterns or get_date_pattern_regexes()

    date_strings = get_date_strings_for_text(text, unique=unique, patterns=patterns)

    for date_string in date_strings:
        try:
            parsed = dateutil_parse(date_string)
            results.append(parsed.date())
        except ParserError:
            continue

    if unique:
        return list(set(results))

    return results
