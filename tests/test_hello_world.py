"""Tests for `hello_world.py`."""

from unittest.mock import patch

from src import found_corner


@patch.object(found_corner, "print")
def test_hello_world(print_mock) -> None:
    found_corner.hello_world()
    print_mock.assert_called_once_with("Hello World!")
