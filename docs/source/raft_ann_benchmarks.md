# RAFT ANN Benchmarks

This project provides a benchmark program for various ANN search implementations. It's especially suitable for comparing GPU implementations as well as comparing GPU against CPU.

## Installing the benchmarks

The easiest way to install these benchmarks is through conda. We suggest using mamba as it generally leads to a faster install time::
```bash
git clone https://github.com/rapidsai/raft.git && cd raft
export RAFT_HOME=$(pwd)

mamba env create --name raft_ann_benchmarks -f conda/environments/bench_ann_cuda-118_arch-x86_64.yaml
conda activate raft_ann_benchmarks

mamba install -c rapidsai -c conda-forge -c nvidia libraft libraft-ann-bench cudatoolkit=11.8*
```
The channel `rapidsai` can easily be substituted `rapidsai-nightly` if nightly benchmarks are desired.

Please see the [build instructions](ann_benchmarks_build.md) to build the benchmarks from source.

## Running the benchmarks

### Usage
There are 4 general steps to running the benchmarks and vizualizing the results:
1. Prepare Dataset
2. Build Index and Search Index
3. Evaluate Results
4. Plot Results

We provide a collection of lightweight Python scripts that are wrappers over
lower level scripts and executables to run our benchmarks. Either Python scripts or
[low-level scripts and executables](ann_benchmarks_low_level.md) are valid methods to run benchmarks,
however plots are only provided through our Python scripts. An environment variable `RAFT_HOME` is
expected to be defined to run these scripts; this variable holds the directory where RAFT is cloned.
### End-to-end example: Million-scale
```bash
export RAFT_HOME=$(pwd)
# All scripts are present in directory raft/bench/ann

# (1) prepare dataset
python bench/ann/get_dataset.py --dataset glove-100-angular --normalize

# (2) build and search index
python bench/ann/run.py --dataset glove-100-inner

# (3) evaluate results
python bench/ann/data_export.py --output out.csv --dataset glove-100-inner

# (4) plot results
python bench/ann/plot.py --result-csv out.csv
```

### End-to-end example: Billion-scale
`bench/ann/get_dataset.py` cannot be used to download the [billion-scale datasets](ann_benchmarks_dataset.md#billion-scale) 
because they are so large. You should instead use our billion-scale datasets guide to download and prepare them.
All other python  mentioned below work as intended once the
billion-scale dataset has been downloaded.
To download Billion-scale datasets, visit [big-ann-benchmarks](http://big-ann-benchmarks.com/neurips21.html)

```bash
export RAFT_HOME=$(pwd)
# All scripts are present in directory raft/bench/ann

mkdir -p data/deep-1B
# (1) prepare dataset
# download manually "Ground Truth" file of "Yandex DEEP"
# suppose the file name is deep_new_groundtruth.public.10K.bin
python bench/ann/split_groundtruth.py --groundtruth data/deep-1B/deep_new_groundtruth.public.10K.bin
# two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced

# (2) build and search index
python bench/ann/run.py --dataset deep-1B

# (3) evaluate results
python bench/ann/data_export.py --output out.csv --dataset deep-1B

# (4) plot results
python bench/ann/plot.py --result-csv out.csv
```

The usage of `bench/ann/split-groundtruth.py` is:
```bash
usage: split_groundtruth.py [-h] --groundtruth GROUNDTRUTH

options:
  -h, --help            show this help message and exit
  --groundtruth GROUNDTRUTH
                        Path to billion-scale dataset groundtruth file (default: None)
```

##### Step 1: Prepare Dataset<a id='prep-dataset'></a>
The script `bench/ann/get_dataset.py` will download and unpack the dataset in directory
that the user provides. As of now, only million-scale datasets are supported by this
script. For more information on [datasets and formats](ann_benchmarks_dataset.md).

The usage of this script is:
```bash
usage: get_dataset.py [-h] [--name NAME] [--dataset-path DATASET_PATH] [--normalize]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset to download (default: glove-100-angular)
  --dataset-path DATASET_PATH
                        path to download dataset (default: ${RAFT_HOME}/bench/ann/data)
  --normalize           normalize cosine distance to inner product (default: False)

When option `normalize` is provided to the script, any dataset that has cosine distances
will be normalized to inner product. So, for example, the dataset `glove-100-angular` 
will be written at location `${RAFT_HOME}/bench/ann/data/glove-100-inner/`.
```

#### Step 2: Build and Search Index
The script `bench/ann/run.py` will build and search indices for a given dataset and its
specified configuration.
To confirgure which algorithms are available, we use `algos.yaml`.
To configure building/searching indices for a dataset, look at [index configuration](#json-index-config).
An entry in `algos.yaml` looks like:
```yaml
raft_ivf_pq:
  executable: RAFT_IVF_PQ_ANN_BENCH
  disabled: false
```
`executable` : specifies the binary that will build/search the index. It is assumed to be
available in `raft/cpp/build/`.
`disabled` : denotes whether an algorithm should be excluded from benchmark runs.

The usage of the script `bench/ann/run.py` is:
```bash
usage: run.py [-h] [--configuration CONFIGURATION] [--dataset DATASET] [--build] [--search] [--algorithms ALGORITHMS] [--indices INDICES] [-f]

options:
usage: run.py [-h] [--configuration CONFIGURATION] [--dataset DATASET] [--dataset-path DATASET_PATH] [--build] [--search] [--algorithms ALGORITHMS] [--indices INDICES] [-f]

options:
  -h, --help            show this help message and exit
  --configuration CONFIGURATION
                        path to configuration file for a dataset (default: None)
  --dataset DATASET     dataset whose configuration file will be used (default: glove-100-inner)
  --dataset-path DATASET_PATH
                        path to dataset folder (default: ${RAFT_HOME}/bench/ann/data)
  --build
  --search
  --algorithms ALGORITHMS
                        run only comma separated list of named algorithms (default: None)
  --indices INDICES     run only comma separated list of named indices. parameter `algorithms` is ignored (default: None)
  -f, --force           re-run algorithms even if their results already exist (default: False)
```
`configuration` and `dataset` : `configuration` is a path to a configuration file for a given dataset.
The configuration file should be name as `<dataset>.json`. It is optional if the name of the dataset is
provided with the `dataset` argument, in which case
a configuration file will be searched for as `${RAFT_HOME}/bench/ann/conf/<dataset>.json`

`dataset-path` : 
1. data is read from `<dataset-path>/<dataset>`
2. indices are built in `<dataset-path>/<dataset>/index`
3. search results are stored in `<dataset-path>/<dataset>/result`

`build` and `search` : if both parameters are not supplied to the script then
it is assumed both are `True`.

`indices` and `algorithms` : these parameters ensure that the algorithm specified for an index 
is available in `algos.yaml` and not disabled, as well as having an associated executable.

#### Step 3: Evaluating Results
The script `bench/ann/data_export.py` will evaluate results for a dataset whose index has been built
and searched with at least one algorithm. For every result file that is available to the script, the output
will be combined and written to a CSV file.

The usage of this script is:
```bash
usage: data_export.py [-h] --output OUTPUT [--recompute] [--dataset DATASET] [--dataset-path DATASET_PATH]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to the CSV output file (default: None)
  --recompute           Recompute metrics (default: False)
  --dataset DATASET     Name of the dataset to export results for (default: glove-100-inner)
  --dataset-path DATASET_PATH
                        path to dataset folder (default: ${RAFT_HOME}/bench/ann/data)
```

#### Step 4: Plot Results
The script `bench/ann/plot.py` will plot all results evaluated to a CSV file for a given dataset.

The usage of this script is:
```bash
usage: plot.py [-h] --result_csv RESULT_CSV [--output OUTPUT] [--x-scale X_SCALE] [--y-scale {linear,log,symlog,logit}] [--raw]

options:
  -h, --help            show this help message and exit
  --result-csv RESULT_CSV
                        Path to CSV Results (default: None)
  --output OUTPUT       Path to the PNG output file (default: ${RAFT_HOME}/out.png)
  --x-scale X_SCALE     Scale to use when drawing the X-axis. Typically linear, logit or a2 (default: linear)
  --y-scale {linear,log,symlog,logit}
                        Scale to use when drawing the Y-axis (default: linear)
  --raw                 Show raw results (not just Pareto frontier) in faded colours (default: False)
```

All algorithms present in the CSV file supplied to this script with parameter `result_csv`
will appear in the plot.

The figure below is the resulting plot of running our benchmarks as of August 2023 for a batch size of 10, on an NVIDIA H100 GPU and an Intel Xeon Platinum 8480CL CPU. It presents the throughput (in Queries-Per-Second) performance for every level of recall.

![Throughput vs recall plot comparing popular ANN algorithms with RAFT's at batch size 10](../../img/raft-vector-search-batch-10.png)

## Adding a new ANN algorithm
### Implementation and Configuration
Implementation of a new algorithm should be a C++ class that inherits `class ANN` (defined in `cpp/bench/ann/src/ann.h`) and implements all the pure virtual functions.

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

<a id='json-index-config'></a>The benchmark program uses JSON configuration file. To add the new algorithm to the benchmark, need be able to specify `build_param`, whose value is a JSON object, and `search_params`, whose value is an array of JSON objects, for this algorithm in configuration file. Still take the configuration for `HnswLib` as an example:
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

### Adding a CMake Target
In `raft/cpp/bench/ann/CMakeLists.txt`, we provide a `CMake` function to configure a new Benchmark target with the following signature:
```
ConfigureAnnBench(
  NAME <algo_name> 
  PATH </path/to/algo/benchmark/source/file> 
  INCLUDES <additional_include_directories> 
  CXXFLAGS <additional_cxx_flags>
  LINKS <additional_link_library_targets>
)
```

To add a target for `HNSWLIB`, we would call the function as:
```
ConfigureAnnBench(
  NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
)
```

This will create an executable called `HNSWLIB_ANN_BENCH`, which can then be used to run `HNSWLIB` benchmarks.
