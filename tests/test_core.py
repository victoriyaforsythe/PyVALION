from PyVALION.library import nearest_element

def test_nearest_element():
    assert nearest_element(np.array([1, 2, 3, 4]), 2.2) == 1
