from my_package import hello


def test_hello() -> None:
    result = hello()
    expected = "Hello from Python-Project-Template!"
    assert result == expected


def test_hello_return_type() -> None:
    result = hello()
    assert isinstance(result, str)
