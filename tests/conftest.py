def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings", "ignore:The NumPy module was reloaded:UserWarning"
    )
