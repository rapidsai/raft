# RAFT ANN Benchmarks

This project provides a benchmark program for various ANN search implementations. It's especially suitable for comparing GPU implementations as well as comparing GPU against CPU.

## Installing the benchmarks

Please see the [build instructions](ann_benchmarks_build.md) to build the benchmarks from source.

## Running the benchmarks

### Usage
There are 4 general steps to running the benchmarks:
1. Prepare Dataset
2. Build Index and Search Index
3. Evaluate Results
4. Plot Results

### Python-based Scripts
We provide a collection of lightweight Python based scripts that are wrappers over
lower level scripts and executables to run our benchmarks. Either Python scripts or
[low-level scripts and executables](ann_benchmarks_low_level.md) are valid methods to run benchmarks,
however plots are only provided through our Python scripts.
#### End-to-end example: Million-scale
```bash
# All scripts are present in directory raft/scripts/ann-benchmarks

# (1) prepare dataset
python scripts/ann-benchmarks/get_dataset.py --name glove-100-angular --normalize

# (2) build and search index
python scripts/ann-benchmarks/run.py --configuration conf/glove-100-inner.json

# (3) evaluate results
python scripts/ann-benchmarks/data_export.py --output out.csv --groundtruth data/glove-100-inner/groundtruth.neighbors.ibin result/glove-100-inner/

# (4) plot results
python scripts/ann-benchmarks/plot.py --result_csv out.csv
```

#### End-to-end example: Billion-scale
The above example does not work at Billion-scale because [data preparation](#prep-dataset) is not yet
supported by `scripts/get_dataset.py`. To download and prepare [billion-scale datasets](ann_benchmarks_low_level.html#billion-scale),
please follow linked section. All other python scripts mentioned below work as intended once the
billion-scale dataset has been downloaded.
To download Billion-scale datasets, visit [big-ann-benchmarks](http://big-ann-benchmarks.com/neurips21.html)

```bash
mkdir -p data/deep-1B && cd data/deep-1B
# (1) prepare dataset
# download manually "Ground Truth" file of "Yandex DEEP"
# suppose the file name is deep_new_groundtruth.public.10K.bin
../../scripts/split_groundtruth.pl deep_new_groundtruth.public.10K.bin groundtruth
# two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced

# (2) build and search index
python scripts/run.py --configuration conf/deep-1B.json

# (3) evaluate results
python scripts/data_export.py --output out.csv --groundtruth data/deep-1B/groundtruth.neighbors.ibin result/deep-1B/

# (4) plot results
python scripts/plot.py --result_csv out.csv
```

##### Step 1: Prepare Dataset<a id='prep-dataset'></a>
The script `scripts/ann-benchmarks/get_dataset.py` will download and unpack the dataset in directory
that the user provides. As of now, only million-scale datasets are supported by this
script. For more information on [datasets and formats](ann_benchmarks_low_level.html#bash-prepare-dataset).

The usage of this script is:
```bash
usage: get_dataset.py [-h] [--name NAME] [--path PATH] [--normalize]

options:
  -h, --help   show this help message and exit
  --name NAME  dataset to download (default: glove-100-angular)
  --path PATH  path to download dataset (default: {os.getcwd()}/data)
  --normalize  normalize cosine distance to inner product (default: False)
```

When option `normalize` is provided to the script, any dataset that has cosine distances
will be normalized to inner product. So, for example, the dataset `glove-100-angular` 
will be written at location `data/glove-100-inner/`.

#### Step 2: Build and Search Index
The script `scripts/ann-benchmarks/run.py` will build and search indices for a given dataset and its
specified configuration.
To confirgure which algorithms are available, we use `algos.yaml`.
To configure building/searching indices for a dataset, look at [index configuration](ann_benchmarks_low_level.html#json-index-config).
An entry in `algos.yaml` looks like:
```yaml
raft_ivf_pq:
  executable: RAFT_IVF_PQ_ANN_BENCH
  disabled: false
```
`executable` : specifies the binary that will build/search the index. It is assumed to be
available in `raft/cpp/build/`.
`disabled` : denotes whether an algorithm should be excluded from benchmark runs.

The usage of the script `scripts/run.py` is:
```bash
usage: run.py [-h] --configuration CONFIGURATION [--algorithms ALGORITHMS] [--indices INDICES] [--force]

options:
  -h, --help            show this help message and exit
  --configuration CONFIGURATION
                        path to configuration file for a dataset (default: None)
  --algorithms ALGORITHMS
                        run only comma separated list of named algorithms (default: None)
  --indices INDICES     run only comma separated list of named indices. parameter `algorithms` is ignored (default: None)
  --force               re-run algorithms even if their results already exist (default: False)
```

Both parameters `indices` and `algorithms` ensure that the algorithm specified for an index 
is available in `algos.yaml` and not disabled, as well as having an associated executable.

#### Step 3: Evaluating Results
The script `scripts/ann-benchmarks/data_export.py` will evaluate results for a dataset whose index has been built
and search with at least one algorithm. For every result file that is supplied to the script, the output
will be combined and written to a CSV file.

The usage of this script is:
```bash
usage: data_export.py [-h] --output OUTPUT [--recompute] --groundtruth GROUNDTRUTH <result_filepaths>

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to the CSV output file (default: None)
  --recompute           Recompute metrics (default: False)
  --groundtruth GROUNDTRUTH
                        Path to groundtruth.neighbors.ibin file for a dataset (default: None)
```

`result_filepaths` : whitespace delimited list of result files/directories that can be capture via pattern match. For more [information and examples](ann_benchmarks_low_level.html#result-filepath-example)

#### Step 4: Plot Results
The script `scripts/ann-benchmarks/plot.py` will plot all results evaluated to a CSV file for a given dataset.

The usage of this script is:
```bash
usage: plot.py [-h] --result_csv RESULT_CSV [--output OUTPUT] [--x-scale X_SCALE] [--y-scale {linear,log,symlog,logit}] [--raw]

options:
  -h, --help            show this help message and exit
  --result_csv RESULT_CSV
                        Path to CSV Results (default: None)
  --output OUTPUT       Path to the PNG output file (default: /home/nfs/dgala/raft/out.png)
  --x-scale X_SCALE     Scale to use when drawing the X-axis. Typically linear, logit or a2 (default: linear)
  --y-scale {linear,log,symlog,logit}
                        Scale to use when drawing the Y-axis (default: linear)
  --raw                 Show raw results (not just Pareto frontier) in faded colours (default: False)
```

All algorithms present in the CSV file supplied to this script with parameter `result_csv`
will appear in the plot.
