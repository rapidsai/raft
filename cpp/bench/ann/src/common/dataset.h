/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>

#ifndef CPU_ONLY
#include <cuda_fp16.h>
#include <raft/util/cudart_utils.hpp>
#else
typedef uint16_t half;
#endif

#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace raft::bench::ann {

// http://big-ann-benchmarks.com/index.html:
// binary format that starts with 8 bytes of data consisting of num_points(uint32_t)
// num_dimensions(uint32) followed by num_pts x num_dimensions x sizeof(type) bytes of
// data stored one vector after another.
// Data files will have suffixes .fbin, .u8bin, and .i8bin to represent float32, uint8
// and int8 type data.
// As extensions for this benchmark, half and int data files will have suffixes .f16bin
// and .ibin, respectively.
template <typename T>
class BinFile {
 public:
  BinFile(const std::string& file,
          const std::string& mode,
          uint32_t subset_first_row = 0,
          uint32_t subset_size      = 0);
  ~BinFile()
  {
    if (fp_) { fclose(fp_); }
  }
  BinFile(const BinFile&)            = delete;
  BinFile& operator=(const BinFile&) = delete;

  void get_shape(size_t* nrows, int* ndims) const
  {
    assert(read_mode_);
    if (!fp_) { open_file_(); }
    *nrows = nrows_;
    *ndims = ndims_;
  }

  void read(T* data) const
  {
    assert(read_mode_);
    if (!fp_) { open_file_(); }
    size_t total = static_cast<size_t>(nrows_) * ndims_;
    if (fread(data, sizeof(T), total, fp_) != total) {
      throw std::runtime_error("fread() BinFile " + file_ + " failed");
    }
  }

  void write(const T* data, uint32_t nrows, uint32_t ndims)
  {
    assert(!read_mode_);
    if (!fp_) { open_file_(); }
    if (fwrite(&nrows, sizeof(uint32_t), 1, fp_) != 1) {
      throw std::runtime_error("fwrite() BinFile " + file_ + " failed");
    }
    if (fwrite(&ndims, sizeof(uint32_t), 1, fp_) != 1) {
      throw std::runtime_error("fwrite() BinFile " + file_ + " failed");
    }

    size_t total = static_cast<size_t>(nrows) * ndims;
    if (fwrite(data, sizeof(T), total, fp_) != total) {
      throw std::runtime_error("fwrite() BinFile " + file_ + " failed");
    }
  }

  T* map() const
  {
    assert(read_mode_);
    if (!fp_) { open_file_(); }
    int fid     = fileno(fp_);
    mapped_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fid, 0);
    if (mapped_ptr_ == MAP_FAILED) {
      throw std::runtime_error("mmap error: Value of errno " + std::to_string(errno) + ", " +
                               std::string(strerror(errno)));
    }
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mapped_ptr_) + 2 * sizeof(uint32_t) +
                                subset_first_row_ * ndims_ * sizeof(T));
  }

  void unmap() const
  {
    if (munmap(mapped_ptr_, file_size_) == -1) {
      throw std::runtime_error("munmap error: " + std::string(strerror(errno)));
    }
  }

 private:
  void check_suffix_();
  void open_file_() const;

  std::string file_;
  bool read_mode_;
  uint32_t subset_first_row_;
  uint32_t subset_size_;

  mutable FILE* fp_;
  mutable uint32_t nrows_;
  mutable uint32_t ndims_;
  mutable size_t file_size_;
  mutable void* mapped_ptr_;
};

template <typename T>
BinFile<T>::BinFile(const std::string& file,
                    const std::string& mode,
                    uint32_t subset_first_row,
                    uint32_t subset_size)
  : file_(file),
    read_mode_(mode == "r"),
    subset_first_row_(subset_first_row),
    subset_size_(subset_size),
    fp_(nullptr)
{
  check_suffix_();

  if (!read_mode_) {
    if (mode == "w") {
      if (subset_first_row != 0) {
        throw std::runtime_error("subset_first_row should be zero for write mode");
      }
      if (subset_size != 0) {
        throw std::runtime_error("subset_size should be zero for write mode");
      }
    } else {
      throw std::runtime_error("BinFile's mode must be either 'r' or 'w': " + file_);
    }
  }
}

template <typename T>
void BinFile<T>::open_file_() const
{
  fp_ = fopen(file_.c_str(), read_mode_ ? "r" : "w");
  if (!fp_) { throw std::runtime_error("open BinFile failed: " + file_); }

  if (read_mode_) {
    struct stat statbuf;
    if (stat(file_.c_str(), &statbuf) != 0) { throw std::runtime_error("stat() failed: " + file_); }
    file_size_ = statbuf.st_size;

    uint32_t header[2];
    if (fread(header, sizeof(uint32_t), 2, fp_) != 2) {
      throw std::runtime_error("read header of BinFile failed: " + file_);
    }
    nrows_ = header[0];
    ndims_ = header[1];

    size_t expected_file_size =
      2 * sizeof(uint32_t) + static_cast<size_t>(nrows_) * ndims_ * sizeof(T);
    if (file_size_ != expected_file_size) {
      throw std::runtime_error("expected file size of " + file_ + " is " +
                               std::to_string(expected_file_size) + ", however, actual size is " +
                               std::to_string(file_size_));
    }

    if (subset_first_row_ >= nrows_) {
      throw std::runtime_error(file_ + ": subset_first_row (" + std::to_string(subset_first_row_) +
                               ") >= nrows (" + std::to_string(nrows_) + ")");
    }
    if (subset_first_row_ + subset_size_ > nrows_) {
      throw std::runtime_error(file_ + ": subset_first_row (" + std::to_string(subset_first_row_) +
                               ") + subset_size (" + std::to_string(subset_size_) + ") > nrows (" +
                               std::to_string(nrows_) + ")");
    }

    if (subset_first_row_) {
      static_assert(sizeof(long) == 8, "fseek() don't support 64-bit offset");
      if (fseek(fp_, sizeof(T) * subset_first_row_ * ndims_, SEEK_CUR) == -1) {
        throw std::runtime_error(file_ + ": fseek failed");
      }
      nrows_ -= subset_first_row_;
    }
    if (subset_size_) { nrows_ = subset_size_; }
  }
}

template <typename T>
void BinFile<T>::check_suffix_()
{
  auto pos = file_.rfind('.');
  if (pos == std::string::npos) {
    throw std::runtime_error("name of BinFile doesn't have a suffix: " + file_);
  }
  std::string suffix = file_.substr(pos + 1);

  if constexpr (std::is_same_v<T, float>) {
    if (suffix != "fbin") {
      throw std::runtime_error("BinFile<float> should has .fbin suffix: " + file_);
    }
  } else if constexpr (std::is_same_v<T, half>) {
    if (suffix != "f16bin") {
      throw std::runtime_error("BinFile<half> should has .f16bin suffix: " + file_);
    }
  } else if constexpr (std::is_same_v<T, int>) {
    if (suffix != "ibin") {
      throw std::runtime_error("BinFile<int> should has .ibin suffix: " + file_);
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    if (suffix != "u8bin") {
      throw std::runtime_error("BinFile<uint8_t> should has .u8bin suffix: " + file_);
    }
  } else if constexpr (std::is_same_v<T, int8_t>) {
    if (suffix != "i8bin") {
      throw std::runtime_error("BinFile<int8_t> should has .i8bin suffix: " + file_);
    }
  } else {
    throw std::runtime_error(
      "T of BinFile<T> should be one of float, half, int, uint8_t, or int8_t");
  }
}

template <typename T>
class Dataset {
 public:
  Dataset(const std::string& name) : name_(name) {}
  Dataset(const std::string& name, const std::string& distance) : name_(name), distance_(distance)
  {
  }
  Dataset(const Dataset&)            = delete;
  Dataset& operator=(const Dataset&) = delete;
  virtual ~Dataset();

  std::string name() const { return name_; }
  std::string distance() const { return distance_; }
  virtual int dim() const               = 0;
  virtual size_t base_set_size() const  = 0;
  virtual size_t query_set_size() const = 0;

  // load data lazily, so don't pay the overhead of reading unneeded set
  // e.g. don't load base set when searching
  const T* base_set() const
  {
    if (!base_set_) { load_base_set_(); }
    return base_set_;
  }

  const T* query_set() const
  {
    if (!query_set_) { load_query_set_(); }
    return query_set_;
  }

  const T* base_set_on_gpu() const;
  const T* query_set_on_gpu() const;
  const T* mapped_base_set() const;

 protected:
  virtual void load_base_set_() const  = 0;
  virtual void load_query_set_() const = 0;
  virtual void map_base_set_() const   = 0;

  std::string name_;
  std::string distance_;

  mutable T* base_set_        = nullptr;
  mutable T* query_set_       = nullptr;
  mutable T* d_base_set_      = nullptr;
  mutable T* d_query_set_     = nullptr;
  mutable T* mapped_base_set_ = nullptr;
};

template <typename T>
Dataset<T>::~Dataset()
{
  delete[] base_set_;
  delete[] query_set_;
#ifndef CPU_ONLY
  if (d_base_set_) { cudaFree(d_base_set_); }
  if (d_query_set_) { cudaFree(d_query_set_); }
#endif
}

template <typename T>
const T* Dataset<T>::base_set_on_gpu() const
{
#ifndef CPU_ONLY
  if (!d_base_set_) {
    base_set();
    RAFT_CUDA_TRY(cudaMalloc((void**)&d_base_set_, base_set_size() * dim() * sizeof(T)));
    RAFT_CUDA_TRY(cudaMemcpy(
      d_base_set_, base_set_, base_set_size() * dim() * sizeof(T), cudaMemcpyHostToDevice));
  }
#endif
  return d_base_set_;
}

template <typename T>
const T* Dataset<T>::query_set_on_gpu() const
{
#ifndef CPU_ONLY
  if (!d_query_set_) {
    query_set();
    RAFT_CUDA_TRY(cudaMalloc((void**)&d_query_set_, query_set_size() * dim() * sizeof(T)));
    RAFT_CUDA_TRY(cudaMemcpy(
      d_query_set_, query_set_, query_set_size() * dim() * sizeof(T), cudaMemcpyHostToDevice));
  }
#endif
  return d_query_set_;
}

template <typename T>
const T* Dataset<T>::mapped_base_set() const
{
  if (!mapped_base_set_) { map_base_set_(); }
  return mapped_base_set_;
}

template <typename T>
class BinDataset : public Dataset<T> {
 public:
  BinDataset(const std::string& name,
             const std::string& base_file,
             size_t subset_first_row,
             size_t subset_size,
             const std::string& query_file,
             const std::string& distance);
  ~BinDataset()
  {
    if (this->mapped_base_set_) { base_file_.unmap(); }
  }

  int dim() const override;
  size_t base_set_size() const override;
  size_t query_set_size() const override;

 private:
  void load_base_set_() const override;
  void load_query_set_() const override;
  void map_base_set_() const override;

  mutable int dim_               = 0;
  mutable size_t base_set_size_  = 0;
  mutable size_t query_set_size_ = 0;

  BinFile<T> base_file_;
  BinFile<T> query_file_;
};

template <typename T>
BinDataset<T>::BinDataset(const std::string& name,
                          const std::string& base_file,
                          size_t subset_first_row,
                          size_t subset_size,
                          const std::string& query_file,
                          const std::string& distance)
  : Dataset<T>(name, distance),
    base_file_(base_file, "r", subset_first_row, subset_size),
    query_file_(query_file, "r")
{
}

template <typename T>
int BinDataset<T>::dim() const
{
  if (dim_ > 0) { return dim_; }
  if (base_set_size() > 0) { return dim_; }
  if (query_set_size() > 0) { return dim_; }
  return dim_;
}

template <typename T>
size_t BinDataset<T>::query_set_size() const
{
  if (query_set_size_ > 0) { return query_set_size_; }
  int dim;
  query_file_.get_shape(&query_set_size_, &dim);
  if (query_set_size_ == 0) { throw std::runtime_error("Zero query set size"); }
  if (dim == 0) { throw std::runtime_error("Zero query set dim"); }
  if (dim_ == 0) {
    dim_ = dim;
  } else if (dim_ != dim) {
    throw std::runtime_error("base set dim (" + std::to_string(dim_) + ") != query set dim (" +
                             std::to_string(dim));
  }
  return query_set_size_;
}

template <typename T>
size_t BinDataset<T>::base_set_size() const
{
  if (base_set_size_ > 0) { return base_set_size_; }
  int dim;
  base_file_.get_shape(&base_set_size_, &dim);
  if (base_set_size_ == 0) { throw std::runtime_error("Zero base set size"); }
  if (dim == 0) { throw std::runtime_error("Zero base set dim"); }
  if (dim_ == 0) {
    dim_ = dim;
  } else if (dim_ != dim) {
    throw std::runtime_error("base set dim (" + std::to_string(dim) + ") != query set dim (" +
                             std::to_string(dim_));
  }
  return base_set_size_;
}

template <typename T>
void BinDataset<T>::load_base_set_() const
{
  this->base_set_ = new T[base_set_size() * dim()];
  base_file_.read(this->base_set_);
}

template <typename T>
void BinDataset<T>::load_query_set_() const
{
  this->query_set_ = new T[query_set_size() * dim()];
  query_file_.read(this->query_set_);
}

template <typename T>
void BinDataset<T>::map_base_set_() const
{
  this->mapped_base_set_ = base_file_.map();
}

}  // namespace  raft::bench::ann
