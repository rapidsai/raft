import numpy as np
from scipy.spatial.distance import cdist

from pylibraft.common import Handle, Stream, device_ndarray
from pylibraft.distance import pairwise_distance


if __name__ == "__main__":
    metric = "euclidean"
    n_rows = 1337
    n_cols = 1337

    input1 = np.random.random_sample((n_rows, n_cols))
    input1 = np.asarray(input1, order="C").astype(np.float64)

    output = np.zeros((n_rows, n_rows), dtype=np.float64)

    expected = cdist(input1, input1, metric)

    expected[expected <= 1e-5] = 0.0

    input1_device = device_ndarray(input1)
    output_device = None

    s2 = Stream()
    handle = Handle(stream=s2)
    ret_output = pairwise_distance(
        input1_device, input1_device, output_device, metric, handle=handle
    )
    handle.sync()

    output_device = ret_output

    actual = output_device.copy_to_host()

    actual[actual <= 1e-5] = 0.0

    assert np.allclose(expected, actual, rtol=1e-4)
