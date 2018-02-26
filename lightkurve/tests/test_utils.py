from __future__ import division, print_function

from numpy.testing import assert_almost_equal
import pytest

from ..utils import module_output_to_channel, channel_to_module_output
from ..utils import running_mean


def test_channel_to_module_output():
    assert channel_to_module_output(1) == (2, 1)
    assert channel_to_module_output(42) == (13, 2)
    assert channel_to_module_output(84) == (24, 4)
    assert channel_to_module_output(33) == (11, 1)
    with pytest.raises(ValueError):
        channel_to_module_output(0)  # Invalid channel


def test_module_output_to_channel():
    assert module_output_to_channel(2, 1) == 1
    assert module_output_to_channel(13, 2) == 42
    assert module_output_to_channel(24, 4) == 84
    assert module_output_to_channel(11, 1) == 33
    with pytest.raises(ValueError):
        module_output_to_channel(0, 1)  # Invalid module
    with pytest.raises(ValueError):
        module_output_to_channel(2, 0)  # Invalid output


def test_running_mean():
    assert_almost_equal(running_mean([1, 2, 3], window_size=1), [1, 2, 3])
    assert_almost_equal(running_mean([1, 2, 3], window_size=2), [1.5, 2.5])
    assert_almost_equal(running_mean([2, 2, 2], window_size=3), [2])
