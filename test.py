import numpy as np
import faiss
rs = np.random.RandomState(123)

xq = rs.rand(50000, 256).astype('float32')
xb = rs.rand(500000, 256).astype('float32')
res = faiss.StandardGpuResources()
quantizer = faiss.IndexFlatL2(256)  # This is an index that performs a brute-force L2 search
index = faiss.IndexIVFFlat(quantizer, 256, 10000, faiss.METRIC_L2)
index.train(xb)
index_gpu_classical = faiss.index_cpu_to_gpu(res, 0, index)
index_gpu_classical.add(xb)
Dref, Iref = index_gpu_classical.search(xq, 10000)
# index = faiss.index_factory(256, "Flat")
index = faiss.IndexIVFFlat(quantizer, 256, 10000, faiss.METRIC_L2)
co = faiss.GpuMultipleClonerOptions()
print(co.use_raft)
co.useFloat16 = True
index_gpu_fp16 = faiss.index_cpu_to_gpu(res, 0, index, co)
index_gpu_fp16.add(xb)
D, I = index_gpu_fp16.search(xq, 100)
(I != Iref).sum() / I.size
# index = faiss.index_factory(256, "Flat")
co = faiss.GpuMultipleClonerOptions()
co.use_raft = True
index_gpu_raft = faiss.index_cpu_to_gpu(res, 0, index, co)
index_gpu_raft.add(xb)
D, I = index_gpu_raft.search(xq, 10)
D, I = index_gpu_raft.search(xq, 200)
print(D, I)
