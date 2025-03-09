import numpy as np
import pytest
import pyplis.helpers as mod  # Import your module as `mod` for easier scaling

# Test: bytescale with default scaling
def test_bytescale_default_scaling():
    img = np.array([[91.06794177, 3.39058326, 84.4221549],
                    [73.88003259, 80.91433048, 4.88878881],
                    [51.53875334, 34.45808177, 27.5873488]])
    expected = np.array([[255, 0, 236],
                         [205, 225, 4],
                         [140, 90, 70]], dtype=np.uint8)
    np.testing.assert_array_equal(mod.bytescale(img), expected)

# Test: bytescale with custom high and low values
def test_bytescale_custom_high_low():
    img = np.array([[91.06794177, 3.39058326, 84.4221549],
                    [73.88003259, 80.91433048, 4.88878881],
                    [51.53875334, 34.45808177, 27.5873488]])
    expected = np.array([[200, 100, 192],
                         [180, 188, 102],
                         [155, 135, 128]], dtype=np.uint8)
    np.testing.assert_array_equal(mod.bytescale(img, high=200, low=100), expected)

# Test: bytescale with custom cmin and cmax values
def test_bytescale_custom_cmin_cmax():
    img = np.array([[91.06794177, 3.39058326, 84.4221549],
                    [73.88003259, 80.91433048, 4.88878881],
                    [51.53875334, 34.45808177, 27.5873488]])
    expected = np.array([[91, 3, 84],
                         [74, 81, 5],
                         [52, 34, 28]], dtype=np.uint8)
    np.testing.assert_array_equal(mod.bytescale(img, cmin=0, cmax=255), expected)

# Test: bytescale with uint8 input
def test_bytescale_uint8_input():
    img = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    # Should return the same array if already uint8
    np.testing.assert_array_equal(mod.bytescale(img), img)

# Test: bytescale with zero range case
def test_bytescale_zero_range():
    img = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
    expected = np.full((3, 3), 0, dtype=np.uint8)
    result = mod.bytescale(img)
    np.testing.assert_array_equal(result, expected)

# Test: bytescale with high less than low
def test_bytescale_high_less_than_low():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="`high` should be larger than `low`."):
        mod.bytescale(img, high=50, low=100)

# Test: bytescale with cmax less than cmin
def test_bytescale_cmax_less_than_cmin():
    img = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="`cmax` should be larger than `cmin`."):
        mod.bytescale(img, cmin=5, cmax=1)

# Placeholder for other function tests (future-proofing)
def test_other_function_example():
    # Example test for a future function in your_module
    assert True  # Replace with actual test when a new function is added
