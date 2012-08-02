class SpykeException(Exception):
    """ Exception thrown when a function in spykeutils encounters a
        problem that is not covered by standard exceptions.

        When using SpykeViewer, these exceptions will be caught and
        shown in the GUI, while general exceptions will not be caught
        (and therefore be visible in the console) for easier
        debugging.
    """
    pass