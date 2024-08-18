import numpy as np
import scipy
from numpy.typing import NDArray


def pulse(freq: float, rate: int, time_start: float = -1, time_stop: float = 1) -> NDArray[np.float32]:
    """Generate gauss pulse signal."""
    duration = time_stop - time_start
    time = np.linspace(time_start, time_stop, int(duration * rate), endpoint=False)
    signal = scipy.signal.gausspulse(time, fc=freq, retquad=False, retenv=False)
    return signal.reshape((1, len(signal)))
