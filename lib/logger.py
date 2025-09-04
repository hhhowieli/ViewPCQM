from typing import Any
from tensorboardX import SummaryWriter
import collections
import numpy as np
import os

class BaseLogger(object):
    def __init__(self, p) -> None:
        self.log_dir = p
        self.log_file = os.path.join(p, "log.txt")

    def log(self, line):
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")


class CSVLogger(BaseLogger):
    def __init__(self, p):
        super().__init__(p)

    def __call__(self, keyword, value):
        pass

class TBLogger(BaseLogger):
    def __init__(self, p) -> None:
        super().__init__(p)
        self.writer = SummaryWriter(p)

    def __call__(self, name, value, n_iter):

        if isinstance(value, (collections.Sequence, np.ndarray)):
            pass
        else:
            self.writer.add_scalar(name, value, n_iter)

