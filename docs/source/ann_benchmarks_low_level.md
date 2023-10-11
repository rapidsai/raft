### Low-level Scripts and Executables
#### End-to-end Example
An end-to-end example (run from the RAFT source code root directory):
```bash
# (1) prepare a dataset
pushd

cd cpp/bench/ann
mkdir data && cd data
wget http://ann-benchmarks.com/glove-100-angular.hdf5

# option -n is used here to normalize vectors so cosine distance is converted
# to inner product; don't use -n for l2 distance
python scripts/hdf5_to_fbin.py -n glove-100-angular.hdf5

mkdir glove-100-inner
mv glove-100-angular.base.fbin glove-100-inner/base.fbin
mv glove-100-angular.query.fbin glove-100-inner/query.fbin
mv glove-100-angular.groundtruth.neighbors.ibin glove-100-inner/groundtruth.neighbors.ibin
mv glove-100-angular.groundtruth.distances.fbin glove-100-inner/groundtruth.distances.fbin
popd

# (2) build index
./cpp/build/RAFT_IVF_FLAT_ANN_BENCH \
  --data_prefix=cpp/bench/ann/data \
  --build \
  --benchmark_filter="raft_ivf_flat\..*" \
  cpp/bench/ann/conf/glove-100-inner.json

# (3) search
./cpp/build/RAFT_IVF_FLAT_ANN_BENCH \
  --data_prefix=cpp/bench/ann/data \
  --benchmark_min_time=2s \
  --benchmark_out=ivf_flat_search.csv \
  --benchmark_out_format=csv \
  --benchmark_counters_tabular \
  --search \
  --benchmark_filter="raft_ivf_flat\..*"
  cpp/bench/ann/conf/glove-100-inner.json

# optional step: plot QPS-Recall figure using data in ivf_flat_search.csv with your favorite tool
```

##### Step 1: Prepare Dataset
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

2. Billion-scale datasets can be found at [`big-ann-benchmarks`](http://big-ann-benchmarks.com). The ground truth file contains both neighbors and distances, thus should be split. A script is provided for this:
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


##### Step 2: Build Index
An index is a data structure to facilitate searching. Different algorithms may use different data structures for their index. We can use `RAFT_IVF_FLAT_ANN_BENCH --build` to build an index and save it to disk.

To run a benchmark executable, like `RAFT_IVF_FLAT_ANN_BENCH`, a JSON configuration file is required. Refer to [`cpp/bench/ann/conf/glove-100-inner.json`](../../cpp/cpp/bench/ann/conf/glove-100-inner.json) as an example. Configuration file has 3 sections:
* `dataset` section specifies the name and files of a dataset, and also the distance in use. Since the `*_ANN_BENCH` programs are for index building and searching, only `base_file` for database vectors and `query_file` for query vectors are needed. Ground truth files are for evaluation thus not needed.
    - To use only a subset of the base dataset, an optional parameter `subset_size` can be specified. It means using only the first `subset_size` vectors of `base_file` as the base dataset.
* `search_basic_param` section specifies basic parameters for searching:
    - `k` is the "k" in "k-nn", that is, the number of neighbors (or results) we want from the searching.
* `index` section specifies an array of configurations for index building and searching:
    - `build_param` and `search_params` are parameters for building and searching, respectively. `search_params` is an array since we will search with different parameters to get different recall values.
    - `file` is the file name of index. Building will save built index to this file, while searching will load this file.
    - if `refine_ratio` is specified, refinement, as a post-processing step of search, will be done. It's for algorithms that compress vectors. For example, if `"refine_ratio" : 2` is set, 2`k` results are first computed, then exact distances of them are computed using original uncompressed vectors, and finally top `k` results among them are kept.


The usage of `*_ANN_BENCH` can be found by running `*_ANN_BENCH --help` on one of the executables:
```bash
$ ./cpp/build/*_ANN_BENCH --help
benchmark [--benchmark_list_tests={true|false}]
          [--benchmark_filter=<regex>]
          [--benchmark_min_time=`<integer>x` OR `<float>s` ]
          [--benchmark_min_warmup_time=<min_warmup_time>]
          [--benchmark_repetitions=<num_repetitions>]
          [--benchmark_enable_random_interleaving={true|false}]
          [--benchmark_report_aggregates_only={true|false}]
          [--benchmark_display_aggregates_only={true|false}]
          [--benchmark_format=<console|json|csv>]
          [--benchmark_out=<filename>]
          [--benchmark_out_format=<json|console|csv>]
          [--benchmark_color={auto|true|false}]
          [--benchmark_counters_tabular={true|false}]
          [--benchmark_context=<key>=<value>,...]
          [--benchmark_time_unit={ns|us|ms|s}]
          [--v=<verbosity>]
          [--build|--search]
          [--overwrite]
          [--data_prefix=<prefix>]
          <conf>.json

Note the non-standard benchmark parameters:
  --build: build mode, will build index
  --search: search mode, will search using the built index
            one and only one of --build and --search should be specified
  --overwrite: force overwriting existing index files
  --data_prefix=<prefix>: prepend <prefix> to dataset file paths specified in the <conf>.json.
  --override_kv=<key:value1:value2:...:valueN>: override a build/search key one or more times multiplying the number of configurations; you can use this parameter multiple times to get the Cartesian product of benchmark configs.
```
* `--build`: build index.
* `--search`: do the searching with built index.
* `--overwrite`: by default, the building mode skips building an index if it find out it already exists. This is useful when adding more configurations to the config; only new indices are build without the need to specify an elaborate filtering regex. By supplying `overwrite` flag, you disable this behavior; all indices are build regardless whether they are already stored on disk.
* `--data_prefix`: prepend an arbitrary path to the data file paths. By default, it is equal to `data`. Note, this does not apply to index file paths.
* `--override_kv`: override a build/search key one or more times multiplying the number of configurations.

In addition to these ANN-specific flags, you can use all of the standard google benchmark flags. Some of the useful flags:
* `--benchmark_filter`: specify subset of benchmarks to run
* `--benchmark_out`, `--benchmark_out_format`: store the output to a file
* `--benchmark_list_tests`: check the available configurations
* `--benchmark_min_time`: specify the minimum duration or number of iterations per case to improve accuracy of the benchmarks.

Refer to the google benchmark [user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md#command-line) for more information about the command-line usage.

##### Step 3: Searching
Use the `--search` flag on any of the `*_ANN_BENCH` executables. Other options are the same as in step 2.

## Adding a new ANN algorithm
Implementation of a new algorithm should be a class that inherits `class ANN` (defined in `cpp/bench/ann/src/ann.h`) and implements all the pure virtual functions.

In addition, it should define two `struct`s for building and searching parameters. The searching parameter class should inherit `struct ANN<T>::AnnSearchParam`. Take `class HnswLib` as an example, its definition is:
```c++
template<typename T>
class HnswLib : public ANN<T> {
public:
  struct BuildParam {
    int M;
    int ef_construction;
    int num_threads;
  };

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    int ef;
    int num_threads;
  };

  // ...
};
```

The benchmark program uses JSON configuration file. To add the new algorithm to the benchmark, need be able to specify `build_param`, whose value is a JSON object, and `search_params`, whose value is an array of JSON objects, for this algorithm in configuration file. Still take the configuration for `HnswLib` as an example:
```json
{
  "name" : "...",
  "algo" : "hnswlib",
  "build_param": {"M":12, "efConstruction":500, "numThreads":32},
  "file" : "/path/to/file",
  "search_params" : [
    {"ef":10, "numThreads":1},
    {"ef":20, "numThreads":1},
    {"ef":40, "numThreads":1}
  ]
},
```

How to interpret these JSON objects is totally left to the implementation and should be specified in `cpp/bench/ann/src/factory.cuh`:
1. First, add two functions for parsing JSON object to `struct BuildParam` and `struct SearchParam`, respectively:
    ```c++
    template<typename T>
    void parse_build_param(const nlohmann::json& conf,
                           typename cuann::HnswLib<T>::BuildParam& param) {
      param.ef_construction = conf.at("efConstruction");
      param.M = conf.at("M");
      if (conf.contains("numThreads")) {
        param.num_threads = conf.at("numThreads");
      }
    }

    template<typename T>
    void parse_search_param(const nlohmann::json& conf,
                            typename cuann::HnswLib<T>::SearchParam& param) {
      param.ef = conf.at("ef");
      if (conf.contains("numThreads")) {
        param.num_threads = conf.at("numThreads");
      }
    }
    ```

2. Next, add corresponding `if` case to functions `create_algo()` and `create_search_param()` by calling parsing functions. The string literal in `if` condition statement must be the same as the value of `algo` in configuration file. For example,
    ```c++
      // JSON configuration file contains a line like:  "algo" : "hnswlib"
      if (algo == "hnswlib") {
         // ...
      }
    ```
