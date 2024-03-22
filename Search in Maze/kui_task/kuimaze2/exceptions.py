class KUIMazeError(BaseException):
    """Base class of all exceptions generated by this module."""
    pass

class NeedsResetError(KUIMazeError):
    """Raised when the problem reached a terminal state and the environment needs to be reset."""
    pass

class ResetImpossibleError(KUIMazeError):
    """Raised when the environment reset is specified in a wrong way such that the new initial state cannot be determined."""
    pass