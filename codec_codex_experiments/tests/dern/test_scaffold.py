def test_dern_package_imports():
    import src.dern as dern
    assert dern.__doc__ is not None
