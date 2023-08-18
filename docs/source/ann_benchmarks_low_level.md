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

##### Step 1: Prepare Dataset <a id='bash-prepare-dataset'></a>
[Instructions](ann_benchmarks_dataset.md)


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
<a id='result-filepath-example'></a>Note that there can be multiple arguments for paths of result files. Each argument can be either a file name or a path. If it's a directory, all files found under it recursively will be used as input files.
An example:
```bash
cpp/bench/ann/scripts/eval.pl groundtruth.neighbors.ibin \
  result/glove-100-angular/10/hnsw/angular_M_24_*.txt \
  result/glove-100-angular/10/faiss/
```
The search result files used by this command are files matching `result/glove-100-angular/10/hnsw/angular_M_24_*.txt`, and all `.txt` files under directory `result/glove-100-angular/10/faiss/` recursively.

This script prints recall and QPS for every result file. Also, it outputs estimated "recall at QPS=2000" and "QPS at recall=0.9", which can be used to compare performance quantitatively.

It saves recall value in result txt file, so avoids to recompute recall if the same command is run again. To force to recompute recall, option `-f` can be used. If option `-o <output.csv>` is specified, a csv output file will be produced. This file can be used to plot Throughput-Recall curves.
