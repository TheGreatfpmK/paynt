import paynt


class TestApiImport:
    def test_get_version_is_exposed_and_callable(self):
        assert hasattr(paynt, "get_version"), "get_version not found in paynt package"
        assert callable(paynt.get_version), "get_version is not callable"

    def test_get_version_returns_a_non_empty_string(self):
        # regression test: get_version() used to crash with AttributeError ('function' object has no
        # attribute '__version__'), since it read version.__version__ instead of calling version() --
        # hasattr/callable alone can't catch this, since the function exists and is callable either way
        assert isinstance(paynt.get_version(), str)
        assert paynt.get_version() != ""
