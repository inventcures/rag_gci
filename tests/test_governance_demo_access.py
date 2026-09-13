"""Short access codes are confined to the synthetic localhost demo."""

import pytest

from clinical_governance.__main__ import demo_access_token


def test_explicit_demo_code_is_used():
    assert demo_access_token("5172") == "5172"


@pytest.mark.parametrize("code", ["123", "1234567890123", "abcd", "１２３４"])
def test_invalid_demo_code_rejected(code):
    with pytest.raises(ValueError):
        demo_access_token(code)


def test_default_remains_random_high_entropy_token():
    first, second = demo_access_token(), demo_access_token()
    assert first != second and len(first) >= 40
