# ANN Benchmarks Datasets

A dataset usually has 4 binary files containing database vectors, query vectors, ground truth neighbors and their corresponding distances. For example, Glove-100 dataset has files `base.fbin` (database vectors), `query.fbin` (query vectors), `groundtruth.neighbors.ibin` (ground truth neighbors), and `groundtruth.distances.fbin` (ground truth distances). The first two files are for index building and searching, while the other two are associated with a particular distance and are used for evaluation.

The file suffixes `.fbin`, `.f16bin`, `.ibin`, `.u8bin`, and `.i8bin` denote that the data type of vectors stored in the file are `float32`, `float16`(a.k.a `half`), `int`, `uint8`, and `int8`, respectively.
These binary files are little-endian and the format is: the first 8 bytes are `num_vectors` (`uint32_t`) and `num_dimensions` (`uint32_t`), and the following `num_vectors * num_dimensions * sizeof(type)` bytes are vectors stored in row-major order.

Some implementation can take `float16` database and query vectors as inputs and will have better performance. Use `script/fbin_to_f16bin.py` to transform dataset from `float32` to `float16` type.

Commonly used datasets can be downloaded from two websites:
1. Million-scale datasets can be found at the [Data sets](https://github.com/erikbern/ann-benchmarks#data-sets) section of [`ann-benchmarks`](https://github.com/erikbern/ann-benchmarks).

    However, these datasets are in HDF5 format. Use `cpp/bench/ann/scripts/hdf5_to_fbin.py` to transform the format. A few Python packages are required to run it:
    ```bash
    pip3 install numpy h5py
    ```
    The usage of this script is:
    ```bash
    $ cpp/bench/ann/scripts/hdf5_to_fbin.py
    usage: scripts/hdf5_to_fbin.py [-n] <input>.hdf5
       -n: normalize base/query set
     outputs: <input>.base.fbin
              <input>.query.fbin
              <input>.groundtruth.neighbors.ibin
              <input>.groundtruth.distances.fbin
    ```
    So for an input `.hdf5` file, four output binary files will be produced. See previous section for an example of prepossessing GloVe dataset.

    Most datasets provided by `ann-benchmarks` use `Angular` or `Euclidean` distance. `Angular` denotes cosine distance. However, computing cosine distance reduces to computing inner product by normalizing vectors beforehand. In practice, we can always do the normalization to decrease computation cost, so it's better to measure the performance of inner product rather than cosine distance. The `-n` option of `hdf5_to_fbin.py` can be used to normalize the dataset.

2. <a id='billion-scale'></a>Billion-scale datasets can be found at [`big-ann-benchmarks`](http://big-ann-benchmarks.com). The ground truth file contains both neighbors and distances, thus should be split. A script is provided for this:
    ```bash
    $ cpp/bench/ann/scripts/split_groundtruth.pl
    usage: script/split_groundtruth.pl input output_prefix
    ```
    Take Deep-1B dataset as an example:
    ```bash
    pushd
    cd cpp/bench/ann
    mkdir -p data/deep-1B && cd data/deep-1B
    # download manually "Ground Truth" file of "Yandex DEEP"
    # suppose the file name is deep_new_groundtruth.public.10K.bin
    ../../scripts/split_groundtruth.pl deep_new_groundtruth.public.10K.bin groundtruth
    # two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced
    popd
    ```
    Besides ground truth files for the whole billion-scale datasets, this site also provides ground truth files for the first 10M or 100M vectors of the base sets. This mean we can use these billion-scale datasets as million-scale datasets. To facilitate this, an optional parameter `subset_size` for dataset can be used. See the next step for further explanation.