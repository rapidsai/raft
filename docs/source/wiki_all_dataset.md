# Wiki-all Dataset

The `wiki-all` dataset was created to stress vector search algorithms at scale with both a large number of vectors and dimensions. The entire dataset contains 88M vectors with 768 dimensions and is meant for testing the types of vectors one would typically encounter in retrieval augmented generation (RAG) workloads. The full dataset is ~251GB in size, which is intentionally larger than the typical memory of GPUs. The massive scale is intended to promote the use of compression and efficient out-of-core methods for both indexing and search.

## Getting the dataset

The dataset is composed of all the available languages of in the [Cohere Wikipedia dataset](https://huggingface.co/datasets/Cohere/wikipedia-22-12). An [English version]( https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) is also available. 


We download the English wiki texts from https://www.kaggle.com/datasets/jjinho/wikipedia-20230701 and multi-lingual wiki texts from Cohere https://huggingface.co/datasets/Cohere/wikipedia-22-12. We notice that the English texts from Cohere is an older and smaller version 2022-12 than the kaggle English wiki texts 2023-07  so we remove the English texts from Cohere completely. In other words, the final wiki texts include English wiki from kaggle and other languages from Cohere. It should be noted that English texts constitute 50% of the total text size. Then, the wiki texts are chunked into 85 million 128-token pieces. For reference, Cohere chunks wiki texts into 104-token pieces. Finally, we compute the embedding of each chunk using paraphrase-multilingual-mpnet-base-v2 embedding model. In the end, we get an embedding matrix of size 85 million by 768.

A version of the dataset is available in the format that can be used directly by the [raft-ann-bench]() tool. It's ~251GB, and has been split into multiple parts.

The following will download all 10 the parts and untar them to a `wiki_all` directory:
```bash
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.{00..9} | tar -xf - -C /datasets/wiki_all/
```

The above has the unfortunate drawback that if the command should fail for any reason, it cannot be restarted. The files can also be downloaded individually and then untarred to the directory. Each file is ~27GB and there are 10 of them.

```bash
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.00
...
curl -s https://data.rapids.ai/raft/datasets/wiki_all/wiki_all.tar.09

cat wiki_all.tar.* | tar -xf - -C /datasets/wiki_all/
```zsx

## Using the dataset

After the dataset is downloaded and extracted to the `wiki_all` directory, the files can be used in the benchmarking tool. The dataset name is `wiki_all`, and the benchmarking tool can be used by specifying `--dataset wiki_all` in the scripts. 