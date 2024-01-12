
__version__ = '0.0.1'
__message__ = 'Version 0.0.1 is full of bugs. Find them out!'


def __get_abspath__():
    import os
    return os.path.abspath(os.path.dirname(__file__))
