# Wiki-all Dataset

The `wiki-all` dataset was created to stress vector search algorithms at scale with both a large number of vectors and dimensions. The entire dataset contains 88M vectors with 768 dimensions and is meant for testing the types of vectors one would typically encounter in retrieval augmented generation (RAG) workloads. The full dataset is ~251GB in size, which is intentionally larger than the typical memory of GPUs. The massive scale is intended to promote the use of compression and efficient out-of-core methods for both indexing and search.

The dataset is composed of English wiki texts from [Kaggle](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) and multi-lingual wiki texts from [Cohere Wikipedia](https://huggingface.co/datasets/Cohere/wikipedia-22-12). 

Cohere's English Texts are older (2022) and smaller than the Kaggle English Wiki texts (2023) so the English texts have been removed from Cohere completely. The final Wiki texts include English Wiki from Kaggle and the other languages from Cohere. The English texts constitute 50% of the total text size. 

To form the final dataset, the Wiki texts were chunked into 85 million 128-token pieces. For reference, Cohere chunks Wiki texts into 104-token pieces. Finally, the embeddings of each chunk were computed using the [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) embedding model. The resulting dataset is an embedding matrix of size 88 million by 768. Also included with the dataset is a query file containing 10k query vectors and a groundtruth file to evaluate nearest neighbors algorithms.

## Getting the dataset

### Full dataset

A version of the dataset is made available in the binary format that can be used directly by the [raft-ann-bench](https://docs.rapids.ai/api/raft/nightly/raft_ann_benchmarks/) tool. The full 88M dataset is ~251GB and the download link below contains tarballs that have been split into multiple parts.

The following will download all 10 the parts and untar them to a `wiki_all_88M` directory:
```bash
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.{00..9} | tar -xf - -C wiki_all_88M/
```

The above has the unfortunate drawback that if the command should fail for any reason, all the parts need to be re-downloaded. The files can also be downloaded individually and then untarred to the directory. Each file is ~27GB and there are 10 of them.

```bash
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.00
...
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.09

cat wiki_all.tar.* | tar -xf - -C wiki_all_88M/
```

### 1M and 10M subsets

Also available are 1M and 10M subsets of the full dataset which are 2.9GB and 29GB, respectively. These subsets also include query sets of 10k vectors and corresponding groundtruth files. 

```bash
curl -s https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar
curl -s https://data.rapids.ai/raft/datasets/wiki_all_10M/wiki_all_10M.tar
```

## Using the dataset

After the dataset is downloaded and extracted to the `wiki_all_88M` directory (or `wiki_all_1M`/`wiki_all_10M` depending on whether the subsets are used), the files can be used in the benchmarking tool. The dataset name is `wiki_all` (or `wiki_all_1M`/`wiki_all_10M`), and the benchmarking tool can be used by specifying the appropriate name `--dataset wiki_all_88M` in the scripts. 

## License info

The English wiki texts available on Kaggle come with the [CC BY-NCSA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license and the Cohere wikipedia data set comes with the [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) license.