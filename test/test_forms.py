import pytest
from pyplis import forms

def test_rectcollection():
    # Initial test in non-optimal design based on former __main__ section in forms.py
    # See PR: https://github.com/jgliss/pyplis/pull/82
    # Related to https://github.com/jgliss/pyplis/issues/29
    rects = {"bla": [1, 2, 3, 4],
             "blub": [10, 20, 30, 40]}

    assert rects["bla"] == [1, 2, 3, 4]
    assert rects["blub"] == [10, 20, 30, 40]
    rect_coll = forms.RectCollection(rects)
    with pytest.raises(AttributeError):
        rect_coll["bla"] = [3, 4, 5, 6]
    assert rect_coll["bla"] == [1, 2, 3, 4]