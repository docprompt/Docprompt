import re
from datetime import date, datetime
from typing import List, Tuple

DateFormatsType = List[Tuple[re.Pattern, str]]

default_date_formats = [
    # Pre-compile regex patterns for efficiency
    # YYYY-MM-DD
    (
        re.compile(r"\b((19|20)\d\d[-](0?[1-9]|1[012])[-](0?[1-9]|[12][0-9]|3[01]))\b"),
        "%Y-%m-%d",
    ),
    # YY-MM-DD
    (
        re.compile(r"\b((\d\d)[-](0?[1-9]|1[012])[-](0?[1-9]|[12][0-9]|3[01]))\b"),
        "%y-%m-%d",
    ),
    # MM-DD-YYYY
    (
        re.compile(r"\b((0?[1-9]|1[012])[-](0?[1-9]|[12][0-9]|3[01])[-](19|20)\d\d)\b"),
        "%m-%d-%Y",
    ),
    # MM-DD-YY
    (
        re.compile(r"\b((0?[1-9]|1[012])[-](0?[1-9]|[12][0-9]|3[01])[-](\d\d))\b"),
        "%m-%d-%y",
    ),
    # DD-MM-YYYY
    (
        re.compile(r"\b((0?[1-9]|[12][0-9]|3[01])[-](0?[1-9]|1[012])[-](19|20)\d\d)\b"),
        "%d-%m-%Y",
    ),
    # DD-MM-YY
    (
        re.compile(r"\b((0?[1-9]|[12][0-9]|3[01])[-](0?[1-9]|1[012])[-](\d\d))\b"),
        "%d-%m-%y",
    ),
    # YYYY/MM/DD
    (
        re.compile(r"\b((19|20)\d\d[/](0?[1-9]|1[012])[/](0?[1-9]|[12][0-9]|3[01]))\b"),
        "%Y/%m/%d",
    ),
    # YY/MM/DD
    (
        re.compile(r"\b((\d\d)[/](0?[1-9]|1[012])[/](0?[1-9]|[12][0-9]|3[01]))\b"),
        "%y/%m/%d",
    ),
    # MM/DD/YYYY
    (
        re.compile(r"\b((0?[1-9]|1[012])[/](0?[1-9]|[12][0-9]|3[01])[/](19|20)\d\d)\b"),
        "%m/%d/%Y",
    ),
    # MM/DD/YY
    (
        re.compile(r"\b((0?[1-9]|1[012])[/](0?[1-9]|[12][0-9]|3[01])[/](\d\d))\b"),
        "%m/%d/%y",
    ),
    # DD/MM/YYYY
    (
        re.compile(r"\b((0?[1-9]|[12][0-9]|3[01])[/](0?[1-9]|1[012])[/](19|20)\d\d)\b"),
        "%d/%m/%Y",
    ),
    # DD/MM/YY
    (
        re.compile(r"\b((0?[1-9]|[12][0-9]|3[01])[/](0?[1-9]|1[012])[/](\d\d))\b"),
        "%d/%m/%y",
    ),
    # YYYY.MM.DD
    (
        re.compile(r"\b((19|20)\d\d[.](0?[1-9]|1[012])[.](0?[1-9]|[12][0-9]|3[01]))\b"),
        "%Y.%m.%d",
    ),
    # YY.MM.DD
    (
        re.compile(r"\b((\d\d)[.](0?[1-9]|1[012])[.](0?[1-9]|[12][0-9]|3[01]))\b"),
        "%y.%m.%d",
    ),
    # MM.DD.YYYY
    (
        re.compile(r"\b((0?[1-9]|1[012])[.](0?[1-9]|[12][0-9]|3[01])[.](19|20)\d\d)\b"),
        "%m.%d.%Y",
    ),
    # MM.DD.YY
    (
        re.compile(r"\b((0?[1-9]|1[012])[.](0?[1-9]|[12][0-9]|3[01])[.](\d\d))\b"),
        "%m.%d.%y",
    ),
    # DD.MM.YYYY
    (
        re.compile(r"\b((0?[1-9]|[12][0-9]|3[01])[.](0?[1-9]|1[012])[.](19|20)\d\d)\b"),
        "%d.%m.%Y",
    ),
    # DD.MM.YY
    (
        re.compile(r"\b((0?[1-9]|[12][0-9]|3[01])[.](0?[1-9]|1[012])[.](\d\d))\b"),
        "%d.%m.%y",
    ),
    # MMMM DDth, YYYY - November 4th, 2023
    (
        re.compile(
            r"\b((January|February|March|April|May|June|July|August|September|October|November|December)\s{1,6}\d{1,2}(st|nd|rd|th)\s{0,2},\s{1,6}\d{4})\b"
        ),
        "%B %d, %Y",
    ),
    # MMMM DD, YYYY - November 4, 2023
    (
        re.compile(
            r"\b((January|February|March|April|May|June|July|August|September|October|November|December)\s{1,6}\d{1,2}\s{0,2},\s{1,6}\d{4})\b"
        ),
        "%B %d, %Y",
    ),
    # MMM DDth, YYYY - Nov 4th, 2023
    (
        re.compile(
            r"\b((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s{1,6}\d{1,2}(st|nd|rd|th)\s{0,2},\s{1,6}\d{4})\b"
        ),
        "%b %d, %Y",
    ),
    # MMM DD, YYYY - Nov 4, 2023
    (
        re.compile(
            r"\b((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s{1,6}\d{1,2}\s{0,2},\s{1,6}\d{4})\b"
        ),
        "%b %d, %Y",
    ),
]


def extract_dates_from_text(
    input_string: str, *, date_formats: DateFormatsType = default_date_formats
) -> List[Tuple[date, str]]:
    """
    Extract dates from a string using a set of predefined regex patterns.

    Returns a list of tuples, where the first element is the date object and the second is the full date string.
    """
    extracted_dates = []

    for regex, date_format in date_formats:
        matches = regex.findall(input_string)

        for match_obj in matches:
            # Extract the full date from the match
            full_date = match_obj[0]  # First group captures the entire date

            if "%d" in date_format:
                parse_date = re.sub(r"(st|nd|rd|th)", "", full_date)
            else:
                parse_date = full_date

            parse_date = re.sub(r"\s+", " ", parse_date).strip()
            parse_date = re.sub(
                r"\s{1,},", ",", parse_date
            ).strip()  # Commas shouldnt have spaces before them

            # Convert to datetime object
            try:
                date_obj = datetime.strptime(parse_date, date_format)
            except ValueError as e:
                print(f"Error parsing date '{full_date}': {e}")
                continue

            extracted_dates.append((date_obj.date(), full_date))

    return extracted_dates
