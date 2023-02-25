# cuANN - CUDA Approximate Nearest Neighbor (ANN) Search

This project provides a benchmark program for various ANN search implementations. It's especially suitable for GPU implementations.

## Developer Guide

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before writing code for this project.

## Benchmark

### Building
Prerequisites for compiling the `benchmark` program:
* CUDA >= 11.3
* GCC >= 8.2 if RAFT is used (preferably GCC 9.5+)
* NCCL >= 2.10 if multi-GPU support is enabled
* FAISS (https://github.com/facebookresearch/faiss) if it's enabled
* cmake in the search path of executable files, for automatically installing FAISS and glog (Google Logging Library, required by GGNN)
* miscellaneous libraries that can be installed using Makefile

#### installing misc. libraries
Most of the libraries are optional, and they can be enabled by the CUANN_USE_XYZ flags in [benchmark/Makefile](benchmark/Makefile). They will be downloadad automatically.


#### installing NCCL
If `CUANN_USE_MULTI_GPU = 1` in `benchmark/Makefile`, NCCL is required.

It's most convenient to install NCCL under `${cuann_bench_path}/third_party/nccl/`, like using "O/S agnostic local installer" downloaded from https://developer.nvidia.com/nccl/nccl-download. Otherwise, may need to modify `CPPFLAGS` and `LDFLAGS` in `benchmark/Makefile` to add include and library paths.


#### installing FAISS library
FAISS can be installed in many ways:
* If `CUANN_USE_RAPIDS_CONTAINER = 1` in `benchmark/Makefile`, RAPIDS Docker container already has FAISS installed
* If `CUANN_USE_RAPIDS_CONTAINER = 0` in `benchmark/Makefile`, FAISS will be installed automatically with the Makefile. However, it reqiures a BLAS implementation is available by setting either environmental paths or Makefile flags.
* Sometimes, FAISS has already been installed in the system and we want to use that. Beside modifying `FAISS_PATH` in `benchmark/Makefile`, we also need to prevent Makefile from installing FAISS again. For that, change the line `faiss: faiss/lib/libfaiss.so` in `third_parth/Makefile` to `faiss:`.

For manual installation: need to install FAISS from source. See [Building from source](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md#building-from-source) for detailed steps.

It's most convenient to install FAISS under `${cuann_bench_path}/third_party/faiss/`. Otherwise, may need to modify `FAISS_PATH` in `benchmark/Makefile`.

An example of cmake build commands:
```
mkdir build && cd build
cmake -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" \
  -DCMAKE_INSTALL_PREFIX=${cuann_bench_path}/third_party/faiss ..
```




#### compiling benchmark
First, modify CUANN_USE_XXX flags at the top of benchmark/Makefile to enable desirable implementations. By default, none is enabled.

Then, just run `cd benchmark && make -j`.

By default, the `benchmark` program accepts dataset of `float` type. To use other type, change the line `using data_t = float;` in `benchmark/src/benchmark.cu` to the target type. For example, `using data_t = uint8_t;` will enable running `benchmark` with dataset of `uint8_t` type.


### Usage
There are 4 steps to run the benchmark:
1. prepare dataset
2. build index
3. search using built index
4. evaluate result

#### TL;DR
A complete example:
```
# (1) prepare a dataset
pip3 install numpy h5py # if they have not been installed already
cd benchmark
mkdir data && cd data
wget http://ann-benchmarks.com/glove-100-angular.hdf5
# option -n is used here to normalize vectors so cosine distance is converted
# to inner product; don't use -n for l2 distance
../../script/hdf5_to_fbin.py -n glove-100-angular.hdf5
mkdir glove-100-inner
mv glove-100-angular.base.fbin glove-100-inner/base.fbin
mv glove-100-angular.query.fbin glove-100-inner/query.fbin
mv glove-100-angular.groundtruth.neighbors.ibin glove-100-inner/groundtruth.neighbors.ibin
mv glove-100-angular.groundtruth.distances.fbin glove-100-inner/groundtruth.distances.fbin
cd ..

# (2) build index
./benchmark -b -i faiss_ivf_flat.nlist1024 conf/glove-100-inner.json

# (3) search
./benchmark -s -i faiss_ivf_flat.nlist1024 conf/glove-100-inner.json

# (4) evaluate result
../script/eval.pl \
  -o result.csv \
  data/glove-100-inner/groundtruth.neighbors.ibin \
  result/glove-100-inner/faiss_ivf_flat

# optional step: plot QPS-Recall figure using data in result.csv with your favorite tool
```


#### step 1: preparing dataset
A dataset usually has 4 binary files containing database vectors, query vectors, ground truth neighbors and their corresponding distances. For example, Glove-100 dataset has files `base.fbin` (database vectors), `query.fbin` (query vectors), `groundtruth.neighbors.ibin` (ground truth neighbors), and `groundtruth.distances.fbin` (ground truth distances). The first two files are for index building and searching, while the other two are associated with a particular distance and are used for evaluation.

The file suffixes `.fbin`, `.f16bin`, `.ibin`, `.u8bin`, and `.i8bin` denote that the data type of vectors stored in the file are `float32`, `float16`(a.k.a `half`), `int`, `uint8`, and `int8`, respectively.
These binary files are little-endian and the format is: the first 8 bytes are `num_vectors` (`uint32_t`) and `num_dimensions` (`uint32_t`), and the following `num_vectors * num_dimensions * sizeof(type)` bytes are vectors stored in row-major order.

Some implementation, like Cagra, can take `float16` database and query vectors as inputs and will have better performance. Use `script/fbin_to_f16bin.py` to transform dataset from `float32` to `float16` type.

Commonly used datasets can be downloaded from two websites:
1.  Million-scale datasets can be found at the [Data sets](https://github.com/erikbern/ann-benchmarks#data-sets) section of [`ann-benchmarks`](https://github.com/erikbern/ann-benchmarks).

    However, these datasets are in HDF5 format. Use `script/hdf5_to_fbin.py` to transform the format. A few Python packages are required to run it:
    ```
    pip3 install numpy h5py
    ```
    The usage of this script is:
    ```
    $ script/hdf5_to_fbin.py
    usage: script/hdf5_to_fbin.py [-n] <input>.hdf5
       -n: normalize base/query set
     outputs: <input>.base.fbin
              <input>.query.fbin
              <input>.groundtruth.neighbors.ibin
              <input>.groundtruth.distances.fbin
    ```
    So for an input `.hdf5` file, four output binary files will be produced. See previous section for an example of prepossessing GloVe dataset.

    Most datasets provided by `ann-benchmarks` use `Angular` or `Euclidean` distance. `Angular` denotes cosine distance. However, computing cosine distance reduces to computing inner product by normalizing vectors beforehand. In practice, we can always do the normalization to decrease computation cost, so it's better to measure the performance of inner product rather than cosine distance. The `-n` option of `hdf5_to_fbin.py` can be used to normalize the dataset.


2.  Billion-scale datasets can be found at [`big-ann-benchmarks`](http://big-ann-benchmarks.com). The ground truth file contains both neighbors and distances, thus should be split. A script is provided for this:
    ```
    $ script/split_groundtruth.pl
    usage: script/split_groundtruth.pl input output_prefix
    ```
    Take Deep-1B dataset as an example:
    ```
    cd benchmark
    mkdir -p data/deep-1B && cd data/deep-1B
    # download manually "Ground Truth" file of "Yandex DEEP"
    # suppose the file name is deep_new_groundtruth.public.10K.bin
    ../../../script/split_groundtruth.pl deep_new_groundtruth.public.10K.bin groundtruth
    # two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced
    ```
    Besides ground truth files for the whole billion-scale datasets, this site also provides ground truth files for the first 10M or 100M vectors of the base sets. This mean we can use these billion-scale datasets as million-scale datasets. To facilitate this, an optional parameter `subset_size` for dataset can be used. See the next step for further explanation.


#### step 2: building index
An index is a data structure to facilitate searching. Different algorithms may use different data structures for their index. We can use `benchmark -b` to build an index and save it to disk.


To run `benchmark`, a JSON configuration file is required. Refer to [`benchmark/conf/glove-100-inner.json`](conf/glove-100-inner.json) as an example. Configuration file has 3 sections:
* `dataset` section specifies the name and files of a dataset, and also the distance in use. Since `benchmark` program is for index building and searching, only `base_file` for database vectors and `query_file` for query vectors are needed. Ground truth files are for evaluation thus not needed.
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



The usage of `benchmark` can be found by running `benchmark -h`:
```
$ ./benchmark -h
usage: ./benchmark -b|s [-f] [-i index_names] conf.json
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
* `-f`: before doing the real task, `benchmark` checks that needed input files exist and output files don't exist. If these conditions are not met, it quits so no file would be overwritten accidentally. To ignore existing output files and force overwrite them, use the `-f` option.
* `-i`: by default, `benchmark -b` will build all indices found in the configuration file, and `benchmark -s` will search using all the indices. To select a subset of indices to build or search, we can use the `-i` option.

It's easier to describe the usage of `-i` option with an example. Suppose we have a configuration file `a.json`, and it contains:
```
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
```
# build all indices: hnsw1, hnsw2 and faiss
./benchmark -b a.json

# build only hnsw1
./benchmark -b -i hnsw1 a.json

# build hnsw1 and hnsw2
./benchmark -b -i hnsw1,hnsw2 a.json

# build hnsw1 and hnsw2
./benchmark -b -i 'hnsw*' a.json

# build hnsw1, hnsw2 and faiss
./benchmark -b -i 'hnsw*,faiss' a.json
```
In the last two commands, we use wildcard "`*`" to match both `hnsw1` and `hnsw2`. Note the use of "`*`" is quite limited. It can occur only at the end of a pattern, so both "`*nsw1`" and "`h*sw1`" are interpreted literally and will not match anything. Also note that quotation marks must be used to prevent "`*`" from being interpreted by the shell.


#### step 3: searching
Use `benchmark -s`. Other options are the same as in step 2.


#### step 4: evaluating results
Use `script/eval.pl` to evaluate benchmark results. The usage is:
```
$ script/eval.pl
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
```
script/eval.pl groundtruth.neighbors.ibin \
  result/glove-100-angular/10/hnsw/angular_M_24_*.txt \
  result/glove-100-angular/10/faiss/
```
The search result files used by this command are files matching `result/glove-100-angular/10/hnsw/angular_M_24_*.txt`, and all `.txt` files under directory `result/glove-100-angular/10/faiss/` recursively.

This script prints recall and QPS for every result file. Also, it outputs estimated "recall at QPS=2000" and "QPS at recall=0.9", which can be used to compare performance quantitatively.

It saves recall value in result txt file, so avoids to recompute recall if the same command is run again. To force to recompute recall, option `-f` can be used. If option `-o <output.csv>` is specified, a csv output file will be produced. This file can be used to plot Throughput-Recall curves.



## How to add a new ANN algorithm
Implementation of a new algorithm should be a class that inherits `class ANN` (defined in `src/cuann_bench/ann.h`) and implements all the pure virtual functions.

In addition, it should define two `struct`s for building and searching parameters. The searching parameter class should inherit `struct ANN<T>::AnnSearchParam`. Take `class HnswLib` as an example, its definition is:
```
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
```
{
  "name" : "...",
  "algo" : "hnswlib",
  "build_param": {"M":12, "efConstruction":500, "numThreads":32},
  "file" : "...",
  "search_params" : [
    {"ef":10, "numThreads":1},
    {"ef":20, "numThreads":1},
    {"ef":40, "numThreads":1},
  ],
  "search_result_file" : "..."
},
```

How to interpret these JSON objects is totally left to the implementation and should be specified in `benchmark/src/factory.cuh`:
* First, add two functions for parsing JSON object to `struct BuildParam` and `struct SearchParam`, respectively:
    ```
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

* Next, add corresponding `if` case to functions `create_algo()` and `create_search_param()` by calling parsing functions. The string literal in `if` condition statement must be the same as the value of `algo` in configuration file. For example,
    ```
      // JSON configuration file contains a line like:  "algo" : "hnswlib"
      if (algo == "hnswlib") {
         // ...
      }
    ```
