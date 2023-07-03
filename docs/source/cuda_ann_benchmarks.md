# CUDA ANN Benchmarks

This project provides a benchmark program for various ANN search implementations. It's especially suitable for comparing GPU implementations as well as comparing GPU against CPU.

## Benchmark

### Dependencies

CUDA 11 and a GPU with Pascal architecture or later are required to run the benchmarks. 

Please refer to the  [installation docs](https://docs.rapids.ai/api/raft/stable/build.html#cuda-gpu-requirements) for the base requirements to build RAFT. 

In addition to the base requirements for building RAFT, additional dependencies needed to build the ANN benchmarks include:
1. FAISS GPU >= 1.7.1
2. Google Logging (GLog)
3. H5Py
4. HNSWLib
5. nlohmann_json
6. GGNN

[rapids-cmake](https://github.com/rapidsai/rapids-cmake) is used to build the ANN benchmarks so the code for dependencies not already supplied in the CUDA toolkit will be downloaded and built automatically.

The easiest (and most reproducible) way to install the dependencies needed to build the ANN benchmarks is to use the conda environment file located in the `conda/environments` directory of the RAFT repository. The following command will use `mamba` (which is preferred over `conda`) to build and activate a new environment for compiling the benchmarks:

```bash
mamba env create --name raft_ann_benchmarks -f conda/environments/bench_ann_cuda-118_arch-x86_64.yaml
conda activate raft_ann_benchmarks
```

The above conda environment will also reduce the compile times as dependencies like FAISS will already be installed and not need to be compiled with `rapids-cmake`.

### Compiling the Benchmarks

After the needed dependencies are satisfied, the easiest way to compile ANN benchmarks is through the `build.sh` script in the root of the RAFT source code repository. The following will build the executables for all the support algorithms:
```bash
./build.sh bench-ann
```

You can limit the algorithms that are built by providing a semicolon-delimited list of executable names (each algorithm is suffixed with `_ANN_BENCH`):
```bash
./build.sh bench-ann -n --limit-bench-ann=HNSWLIB_ANN_BENCH;RAFT_IVF_PQ_ANN_BENCH
```

Available targets to use with `--limit-bench-ann` are:
- FAISS_IVF_FLAT_ANN_BENCH
- FAISS_IVF_PQ_ANN_BENCH
- FAISS_BFKNN_ANN_BENCH
- GGNN_ANN_BENCH
- HNSWLIB_ANN_BENCH
- RAFT_CAGRA_ANN_BENCH
- RAFT_IVF_PQ_ANN_BENCH
- RAFT_IVF_FLAT_ANN_BENCH

By default, the `*_ANN_BENCH` executables program infer the dataset's datatype from the filename's extension. For example, an extension of `fbin` uses a `float` datatype, `f16bin` uses a `float16` datatype, extension of `i8bin` uses `int8_t` datatype, and `u8bin` uses `uint8_t` type. Currently, only `float`, `float16`, int8_t`, and `unit8_t` are supported.

### Usage
There are 4 general steps to running the benchmarks:
1. Prepare Dataset
2. Build Index
3. Search Using Built Index
4. Evaluate Result

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
./cpp/build/RAFT_IVF_FLAT_ANN_BENCH -b -i raft_ivf_flat.nlist1024 conf/glove-100-inner.json

# (3) search
./cpp/build/RAFT_IVF_FLAT_ANN_BENCH -s -i raft_ivf_flat.nlist1024 conf/glove-100-inner.json

# (4) evaluate result
pushd
cd cpp/bench/ann
./scripts/eval.pl \
  -o result.csv \
  data/glove-100-inner/groundtruth.neighbors.ibin \
  result/glove-100-inner/faiss_ivf_flat
popd 

# optional step: plot QPS-Recall figure using data in result.csv with your favorite tool
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
An index is a data structure to facilitate searching. Different algorithms may use different data structures for their index. We can use `RAFT_IVF_FLAT_ANN_BENCH -b` to build an index and save it to disk.

To run a benchmark executable, like `RAFT_IVF_FLAT_ANN_BENCH`, a JSON configuration file is required. Refer to [`cpp/bench/ann/conf/glove-100-inner.json`](../../cpp/cpp/bench/ann/conf/glove-100-inner.json) as an example. Configuration file has 3 sections:
* `dataset` section specifies the name and files of a dataset, and also the distance in use. Since the `*_ANN_BENCH` programs are for index building and searching, only `base_file` for database vectors and `query_file` for query vectors are needed. Ground truth files are for evaluation thus not needed.
    - To use only a subset of the base dataset, an optional parameter `subset_size` can be specified. It means using only the first `subset_size` vectors of `base_file` as the base dataset.
* `search_basic_param` section specifies basic parameters for searching:
    - `k` is the "k" in "k-nn", that is, the number of neighbors (or results) we want from the searching.
    -  `run_count` means how many times we run the searching. A single run of searching will search neighbors for all vectors in `test` set. The total time used for a run is recorded, and the final searching time is the smallest one among these runs.
* `index` section specifies an array of configurations for index building and searching:
    - `build_param` and `search_params` are parameters for building and searching, respectively. `search_params` is an array since we will search with different parameters to get different recall values.
    - `file` is the file name of index. Building will save built index to this file, while searching will load this file.
    - `search_result_file` is the file name prefix of searching results. Searching will save results to these files, and plotting script will read these files to plot results. Note this is a prefix rather than a whole file name. Suppose its value is `${prefix}`, then the real file names are like `${prefix}.0.{ibin|txt}`, `${prefix}.1.{ibin|txt}`, etc. Each of them corresponds to an item in `search_params` array. That is, for one searching parameter, there will be some corresponding search result files.
    - if `multigpu` is specified, multiple GPUs will be used for index build and search.
    - if `refine_ratio` is specified, refinement, as a post-processing step of search, will be done. It's for algorithms that compress vectors. For example, if `"refine_ratio" : 2` is set, 2`k` results are first computed, then exact distances of them are computed using original uncompressed vectors, and finally top `k` results among them are kept.


The usage of `*_ANN_BENCH` can be found by running `*_ANN_BENCH -h` on one of the executables:
```bash
$ ./cpp/build/*_ANN_BENCH -h
usage: ./cpp/build/*_ANN_BENCH -b|s [-f] [-i index_names] conf.json
   -b: build mode, will build index
   -s: search mode, will search using built index
       one and only one of -b and -s should be specified
   -f: force overwriting existing output files
   -i: by default will build/search all the indices found in conf.json
       '-i' can be used to select a subset of indices
       'index_names' is a list of comma-separated index names
       '*' is allowed as the last character of a name to select all matched indices
       for example, -i "hnsw1,hnsw2,faiss" or -i "hnsw*,faiss"
```
* `-b`: build index.
* `-s`: do the searching with built index.
* `-f`: before doing the real task, the program checks that needed input files exist and output files don't exist. If these conditions are not met, it quits so no file would be overwritten accidentally. To ignore existing output files and force overwrite them, use the `-f` option.
* `-i`: by default, the `-b` flag will build all indices found in the configuration file, and `-s` will search using all the indices. To select a subset of indices to build or search, we can use the `-i` option.

It's easier to describe the usage of `-i` option with an example. Suppose we have a configuration file `a.json`, and it contains:
```json
  "index" : [
    {
      "name" : "hnsw1",
      ...
    },
    {
      "name" : "hnsw1",
      ...
    },
    {
      "name" : "faiss",
      ...
    }
  ]
```
Then,
```bash
# build all indices: hnsw1, hnsw2 and faiss
./cpp/build/HNSWLIB_ANN_BENCH -b a.json

# build only hnsw1
./cpp/build/HNSWLIB_ANN_BENCH -b -i hnsw1 a.json

# build hnsw1 and hnsw2
./cpp/build/HNSWLIB_ANN_BENCH -b -i hnsw1,hnsw2 a.json

# build hnsw1 and hnsw2
./cpp/build/HNSWLIB_ANN_BENCH -b -i 'hnsw*' a.json

# build faiss
./cpp/build/FAISS_IVF_FLAT_ANN_BENCH -b -i 'faiss' a.json
```
In the last two commands, we use wildcard "`*`" to match both `hnsw1` and `hnsw2`. Note the use of "`*`" is quite limited. It can occur only at the end of a pattern, so both "`*nsw1`" and "`h*sw1`" are interpreted literally and will not match anything. Also note that quotation marks must be used to prevent "`*`" from being interpreted by the shell.


##### Step 3: Searching
Use the `-s` flag on any of the `*_ANN_BENCH` executables. Other options are the same as in step 2.


##### Step 4: Evaluating Results
Use `cpp/bench/ann/scripts/eval.pl` to evaluate benchmark results. The usage is:
```bash
$ cpp/bench/ann/scripts/eval.pl
usage: [-f] [-o output.csv] groundtruth.neighbors.ibin result_paths...
  result_paths... are paths to the search result files.
    Can specify multiple paths.
    For each of them, if it's a directory, all the .txt files found under
    it recursively will be regarded as inputs.

  -f: force to recompute recall and update it in result file if needed
  -o: also write result to a csv file
```
Note that there can be multiple arguments for paths of result files. Each argument can be either a file name or a path. If it's a directory, all files found under it recursively will be used as input files.
An example:
```bash
cpp/bench/ann/scripts/eval.pl groundtruth.neighbors.ibin \
  result/glove-100-angular/10/hnsw/angular_M_24_*.txt \
  result/glove-100-angular/10/faiss/
```
The search result files used by this command are files matching `result/glove-100-angular/10/hnsw/angular_M_24_*.txt`, and all `.txt` files under directory `result/glove-100-angular/10/faiss/` recursively.

This script prints recall and QPS for every result file. Also, it outputs estimated "recall at QPS=2000" and "QPS at recall=0.9", which can be used to compare performance quantitatively.

It saves recall value in result txt file, so avoids to recompute recall if the same command is run again. To force to recompute recall, option `-f` can be used. If option `-o <output.csv>` is specified, a csv output file will be produced. This file can be used to plot Throughput-Recall curves.

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
    {"ef":40, "numThreads":1},
  ],
  "search_result_file" : "/path/to/file"
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
