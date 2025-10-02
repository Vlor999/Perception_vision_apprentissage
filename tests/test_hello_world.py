"""Tests for `hello_world.py`."""

from unittest.mock import patch

from src import hello


@patch.object(hello, "print")
def test_hello_world(print_mock) -> None:
    hello.hello_world()
    print_mock.assert_called_once_with("Hello World!")
