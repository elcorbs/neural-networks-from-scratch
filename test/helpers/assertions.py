from numpy import isclose

def listEquals(actual, expected):
    assert len(actual) == len(expected)
    assert all([isclose(a, b) for a, b in zip(actual, expected)])

def arrayEqual(actual, expected):
    assert len(actual) == len(expected)
    [listEquals(a, b) for a, b in zip(actual, expected)]