from metrics import cer, edit_distance


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3


def test_cer():
    assert cer("abc", "abc") == 0.0
    assert cer("axc", "abc") == 1 / 3

