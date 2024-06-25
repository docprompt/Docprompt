from datetime import date

from docprompt.utils.date_extraction import extract_dates_from_text

STRING_A = """
There was a meeting on 2021-01-01 and another on 2021-01-02.

Meanwhile on September 1, 2021, there was a third meeting.

The final meeting was on 4/5/2021.
"""


def test_date_extraction():
    dates = extract_dates_from_text(STRING_A)

    assert len(dates) == 5

    dates.sort(key=lambda x: x[0])

    assert dates[0][0] == date(2021, 1, 1)
    assert dates[0][1] == "2021-01-01"

    assert dates[1][0] == date(2021, 1, 2)
    assert dates[1][1] == "2021-01-02"

    assert dates[2][0] == date(2021, 4, 5)
    assert dates[2][1] == "4/5/2021"

    assert dates[3][0] == date(2021, 5, 4)
    assert dates[3][1] == "4/5/2021"

    assert dates[4][0] == date(2021, 9, 1)
    assert dates[4][1] == "September 1, 2021"
