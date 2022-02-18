from abc import ABC, abstractmethod
import os


class SuperessVerbosity(object):

    """For custom estimators that just won't be quiet..."""

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class ModelBase(ABC):

    """Model pipeline base class"""

    @abstractmethod
    def build():
        """Construct model"""
        pass

    @abstractmethod
    def preprocess():
        """Construct preprocessing pipeline for model"""
        pass

    @abstractmethod
    def params():
        """Construct hyperparameters"""
        pass

    @abstractmethod
    def fit_params():
        """Construct model fit parameters"""
        pass


if __name__ == "__main__":
    pass
