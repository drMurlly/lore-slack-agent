import conduit


def test_version():
    assert conduit.__version__ == "0.1.0"


def test_imports():
    import importlib
    assert importlib.import_module("conduit") is conduit
