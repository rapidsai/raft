# raft 21.12.00 (9 Dec 2021)

## 🚨 Breaking Changes

- Use 64 bit CuSolver API for Eigen decomposition ([#349](https://github.com/rapidsai/raft/pull/349)) [@lowener](https://github.com/lowener)

## 🐛 Bug Fixes

- Fixing bad host-&gt;device copy ([#375](https://github.com/rapidsai/raft/pull/375)) [@cjnolet](https://github.com/cjnolet)
- Fix coalesced access checks in matrix_vector_op ([#372](https://github.com/rapidsai/raft/pull/372)) [@achirkin](https://github.com/achirkin)
- Port libcudacxx patch from cudf ([#370](https://github.com/rapidsai/raft/pull/370)) [@dantegd](https://github.com/dantegd)
- Fixing overflow in expanded distances ([#365](https://github.com/rapidsai/raft/pull/365)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- Getting doxygen to run ([#371](https://github.com/rapidsai/raft/pull/371)) [@cjnolet](https://github.com/cjnolet)

## 🛠️ Improvements

- Upgrade `clang` to `11.1.0` ([#394](https://github.com/rapidsai/raft/pull/394)) [@galipremsagar](https://github.com/galipremsagar)
- Fix Changelog Merge Conflicts for `branch-21.12` ([#390](https://github.com/rapidsai/raft/pull/390)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin max `dask` &amp; `distributed` ([#388](https://github.com/rapidsai/raft/pull/388)) [@galipremsagar](https://github.com/galipremsagar)
- Removing conflict w/ CUDA_CHECK ([#378](https://github.com/rapidsai/raft/pull/378)) [@cjnolet](https://github.com/cjnolet)
- Update RAFT test directory ([#359](https://github.com/rapidsai/raft/pull/359)) [@viclafargue](https://github.com/viclafargue)
- Update to UCX-Py 0.23 ([#358](https://github.com/rapidsai/raft/pull/358)) [@pentschev](https://github.com/pentschev)
- Hiding implementation details for random, stats, and matrix ([#356](https://github.com/rapidsai/raft/pull/356)) [@divyegala](https://github.com/divyegala)
- README updates ([#351](https://github.com/rapidsai/raft/pull/351)) [@cjnolet](https://github.com/cjnolet)
- Use 64 bit CuSolver API for Eigen decomposition ([#349](https://github.com/rapidsai/raft/pull/349)) [@lowener](https://github.com/lowener)
- Hiding implementation details for distance primitives (dense + sparse) ([#344](https://github.com/rapidsai/raft/pull/344)) [@cjnolet](https://github.com/cjnolet)
- Unpin `dask` &amp; `distributed` in CI ([#338](https://github.com/rapidsai/raft/pull/338)) [@galipremsagar](https://github.com/galipremsagar)

# raft 21.10.00 (7 Oct 2021)

## 🚨 Breaking Changes

- Miscellaneous tech debts/cleanups ([#286](https://github.com/rapidsai/raft/pull/286)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Accounting for rmm::cuda_stream_pool not having a constructor for 0 streams ([#329](https://github.com/rapidsai/raft/pull/329)) [@divyegala](https://github.com/divyegala)
- Fix wrong lda parameter in gemv ([#327](https://github.com/rapidsai/raft/pull/327)) [@achirkin](https://github.com/achirkin)
- Fix `matrixVectorOp` to verify promoted pointer type is still aligned to vectorized load boundary ([#325](https://github.com/rapidsai/raft/pull/325)) [@viclafargue](https://github.com/viclafargue)
- Pin rmm to branch-21.10 and remove warnings from kmeans.hpp ([#322](https://github.com/rapidsai/raft/pull/322)) [@dantegd](https://github.com/dantegd)
- Temporarily pin RMM while refactor removes deprecated calls ([#315](https://github.com/rapidsai/raft/pull/315)) [@dantegd](https://github.com/dantegd)
- Fix more warnings ([#311](https://github.com/rapidsai/raft/pull/311)) [@harrism](https://github.com/harrism)

## 📖 Documentation

- Fix build doc ([#316](https://github.com/rapidsai/raft/pull/316)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- Add Hamming, Jensen-Shannon, KL-Divergence, Russell rao and Correlation distance metrics support ([#306](https://github.com/rapidsai/raft/pull/306)) [@mdoijade](https://github.com/mdoijade)

## 🛠️ Improvements

- Pin max `dask` and `distributed` versions to `2021.09.1` ([#334](https://github.com/rapidsai/raft/pull/334)) [@galipremsagar](https://github.com/galipremsagar)
- Make sure we keep the rapids-cmake and raft cal version in sync ([#331](https://github.com/rapidsai/raft/pull/331)) [@robertmaynard](https://github.com/robertmaynard)
- Add broadcast with const input iterator ([#328](https://github.com/rapidsai/raft/pull/328)) [@seunghwak](https://github.com/seunghwak)
- Fused L2 (unexpanded) kNN kernel for NN &lt;= 64, without using temporary gmem to store intermediate distances ([#324](https://github.com/rapidsai/raft/pull/324)) [@mdoijade](https://github.com/mdoijade)
- Update with rapids cmake new features ([#320](https://github.com/rapidsai/raft/pull/320)) [@robertmaynard](https://github.com/robertmaynard)
- Update to UCX-Py 0.22 ([#319](https://github.com/rapidsai/raft/pull/319)) [@pentschev](https://github.com/pentschev)
- Fix Forward-Merge Conflicts ([#318](https://github.com/rapidsai/raft/pull/318)) [@ajschmidt8](https://github.com/ajschmidt8)
- Enable CUDA device code warnings as errors ([#307](https://github.com/rapidsai/raft/pull/307)) [@harrism](https://github.com/harrism)
- Remove max version pin for dask &amp; distributed on development branch ([#303](https://github.com/rapidsai/raft/pull/303)) [@galipremsagar](https://github.com/galipremsagar)
- Warnings are errors ([#299](https://github.com/rapidsai/raft/pull/299)) [@harrism](https://github.com/harrism)
- Use the new RAPIDS.cmake to fetch rapids-cmake ([#298](https://github.com/rapidsai/raft/pull/298)) [@robertmaynard](https://github.com/robertmaynard)
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#295](https://github.com/rapidsai/raft/pull/295)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Miscellaneous tech debts/cleanups ([#286](https://github.com/rapidsai/raft/pull/286)) [@viclafargue](https://github.com/viclafargue)
- Random Ball Cover Algorithm for 2D Haversine/Euclidean ([#213](https://github.com/rapidsai/raft/pull/213)) [@cjnolet](https://github.com/cjnolet)

# raft 21.08.00 (4 Aug 2021)

## 🚨 Breaking Changes

- expose epsilon parameter to allow precision to to be specified ([#275](https://github.com/rapidsai/raft/pull/275)) [@ChuckHastings](https://github.com/ChuckHastings)

## 🐛 Bug Fixes

- Fix support for different input and output types in linalg::reduce ([#296](https://github.com/rapidsai/raft/pull/296)) [@Nyrio](https://github.com/Nyrio)
- Const raft handle in sparse bfknn ([#280](https://github.com/rapidsai/raft/pull/280)) [@cjnolet](https://github.com/cjnolet)
- Add `cuco::cuco` to list of linked libraries ([#279](https://github.com/rapidsai/raft/pull/279)) [@trxcllnt](https://github.com/trxcllnt)
- Use nested include in destination of install headers to avoid docker permission issues ([#263](https://github.com/rapidsai/raft/pull/263)) [@dantegd](https://github.com/dantegd)
- Update UCX-Py version to 0.21 ([#255](https://github.com/rapidsai/raft/pull/255)) [@pentschev](https://github.com/pentschev)
- Fix mst knn test build failure due to RMM device_buffer change ([#253](https://github.com/rapidsai/raft/pull/253)) [@mdoijade](https://github.com/mdoijade)

## 🚀 New Features

- Add chebyshev, canberra, minkowksi and hellinger distance metrics ([#276](https://github.com/rapidsai/raft/pull/276)) [@mdoijade](https://github.com/mdoijade)
- Move FAISS ANN wrappers to RAFT ([#265](https://github.com/rapidsai/raft/pull/265)) [@cjnolet](https://github.com/cjnolet)
- Remaining sparse semiring distances ([#261](https://github.com/rapidsai/raft/pull/261)) [@cjnolet](https://github.com/cjnolet)
- removing divye from codeowners ([#257](https://github.com/rapidsai/raft/pull/257)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Pinning cuco to a specific commit hash for release ([#304](https://github.com/rapidsai/raft/pull/304)) [@rlratzel](https://github.com/rlratzel)
- Pin max `dask` &amp; `distributed` versions ([#301](https://github.com/rapidsai/raft/pull/301)) [@galipremsagar](https://github.com/galipremsagar)
- Overlap epilog compute with ldg of next grid stride in pairwise distance &amp; fusedL2NN kernels ([#292](https://github.com/rapidsai/raft/pull/292)) [@mdoijade](https://github.com/mdoijade)
- Always add faiss library alias if it&#39;s missing ([#287](https://github.com/rapidsai/raft/pull/287)) [@trxcllnt](https://github.com/trxcllnt)
- Use `NVIDIA/cuCollections` repo again ([#284](https://github.com/rapidsai/raft/pull/284)) [@trxcllnt](https://github.com/trxcllnt)
- Use the 21.08 branch of rapids-cmake as rmm requires it ([#278](https://github.com/rapidsai/raft/pull/278)) [@robertmaynard](https://github.com/robertmaynard)
- expose epsilon parameter to allow precision to to be specified ([#275](https://github.com/rapidsai/raft/pull/275)) [@ChuckHastings](https://github.com/ChuckHastings)
- Fix `21.08` forward-merge conflicts ([#274](https://github.com/rapidsai/raft/pull/274)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add lds and sts inline ptx instructions to force vector instruction generation ([#273](https://github.com/rapidsai/raft/pull/273)) [@mdoijade](https://github.com/mdoijade)
- Move ANN to RAFT (additional updates) ([#270](https://github.com/rapidsai/raft/pull/270)) [@cjnolet](https://github.com/cjnolet)
- Sparse semirings cleanup + hash table &amp; batching strategies ([#269](https://github.com/rapidsai/raft/pull/269)) [@divyegala](https://github.com/divyegala)
- Revert &quot;pin dask versions in CI ([#260)&quot; (#264](https://github.com/rapidsai/raft/pull/260)&quot; (#264)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pass stream to device_scalar::value() calls. ([#259](https://github.com/rapidsai/raft/pull/259)) [@harrism](https://github.com/harrism)
- Update get_rmm.cmake to better support CalVer ([#258](https://github.com/rapidsai/raft/pull/258)) [@harrism](https://github.com/harrism)
- Add Grid stride pairwise dist and fused L2 NN kernels ([#250](https://github.com/rapidsai/raft/pull/250)) [@mdoijade](https://github.com/mdoijade)
- Fix merge conflicts ([#236](https://github.com/rapidsai/raft/pull/236)) [@ajschmidt8](https://github.com/ajschmidt8)

# raft 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- Update UCX-Py version to 0.20 ([#254](https://github.com/rapidsai/raft/pull/254)) [@pentschev](https://github.com/pentschev)
- cuco git tag update (again) ([#248](https://github.com/rapidsai/raft/pull/248)) [@seunghwak](https://github.com/seunghwak)
- Revert PR #232 for 21.06 release ([#246](https://github.com/rapidsai/raft/pull/246)) [@dantegd](https://github.com/dantegd)
- Python comms to hold onto server endpoints ([#241](https://github.com/rapidsai/raft/pull/241)) [@cjnolet](https://github.com/cjnolet)
- Fix Thrust 1.12 compile errors ([#231](https://github.com/rapidsai/raft/pull/231)) [@trxcllnt](https://github.com/trxcllnt)
- Make sure we use CalVer when checking out rapids-cmake ([#230](https://github.com/rapidsai/raft/pull/230)) [@robertmaynard](https://github.com/robertmaynard)
- Loss of Precision in MST weight alteration ([#223](https://github.com/rapidsai/raft/pull/223)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- cuco git tag update ([#243](https://github.com/rapidsai/raft/pull/243)) [@seunghwak](https://github.com/seunghwak)
- Update `CHANGELOG.md` links for calver ([#233](https://github.com/rapidsai/raft/pull/233)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add Grid stride pairwise dist and fused L2 NN kernels ([#232](https://github.com/rapidsai/raft/pull/232)) [@mdoijade](https://github.com/mdoijade)
- Updates to enable HDBSCAN ([#208](https://github.com/rapidsai/raft/pull/208)) [@cjnolet](https://github.com/cjnolet)

# raft 0.19.0 (21 Apr 2021)

## 🐛 Bug Fixes

- Exposing spectral random seed property ([#193](https://github.com//rapidsai/raft/pull/193)) [@cjnolet](https://github.com/cjnolet)
- Fix pointer arithmetic in spmv smem kernel ([#183](https://github.com//rapidsai/raft/pull/183)) [@lowener](https://github.com/lowener)
- Modify default value for rowMajorIndex and rowMajorQuery in bf-knn ([#173](https://github.com//rapidsai/raft/pull/173)) [@viclafargue](https://github.com/viclafargue)
- Remove setCudaMallocWarning() call for libfaiss[@v1.7.0 ([#167](https://github.com//rapidsai/raft/pull/167)) @trxcllnt](https://github.com/v1.7.0 ([#167](https://github.com//rapidsai/raft/pull/167)) @trxcllnt)
- Add const to KNN handle ([#157](https://github.com//rapidsai/raft/pull/157)) [@hlinsen](https://github.com/hlinsen)

## 🚀 New Features

- Moving optimized L2 1-nearest neighbors implementation from cuml ([#158](https://github.com//rapidsai/raft/pull/158)) [@cjnolet](https://github.com/cjnolet)

## 🛠️ Improvements

- Fixing codeowners ([#194](https://github.com//rapidsai/raft/pull/194)) [@cjnolet](https://github.com/cjnolet)
- Adjust Hellinger pairwise distance to vaoid NaNs ([#189](https://github.com//rapidsai/raft/pull/189)) [@lowener](https://github.com/lowener)
- Add column major input support in contractions_nt kernels with new kernel policy for it ([#188](https://github.com//rapidsai/raft/pull/188)) [@mdoijade](https://github.com/mdoijade)
- Dice formula correction ([#186](https://github.com//rapidsai/raft/pull/186)) [@lowener](https://github.com/lowener)
- Scaling knn graph fix connectivities algorithm ([#181](https://github.com//rapidsai/raft/pull/181)) [@cjnolet](https://github.com/cjnolet)
- Fixing RAFT CI &amp; a few small updates for SLHC Python wrapper ([#178](https://github.com//rapidsai/raft/pull/178)) [@cjnolet](https://github.com/cjnolet)
- Add Precomputed to the DistanceType enum (for cuML DBSCAN) ([#177](https://github.com//rapidsai/raft/pull/177)) [@Nyrio](https://github.com/Nyrio)
- Enable matrix::copyRows for row major input ([#176](https://github.com//rapidsai/raft/pull/176)) [@tfeher](https://github.com/tfeher)
- Add Dice distance to distancetype enum ([#174](https://github.com//rapidsai/raft/pull/174)) [@lowener](https://github.com/lowener)
- Porting over recent updates to distance prim from cuml ([#172](https://github.com//rapidsai/raft/pull/172)) [@cjnolet](https://github.com/cjnolet)
- Update KNN ([#171](https://github.com//rapidsai/raft/pull/171)) [@viclafargue](https://github.com/viclafargue)
- Adding translations parameter to brute_force_knn ([#170](https://github.com//rapidsai/raft/pull/170)) [@viclafargue](https://github.com/viclafargue)
- Update Changelog Link ([#169](https://github.com//rapidsai/raft/pull/169)) [@ajschmidt8](https://github.com/ajschmidt8)
- Map operation ([#168](https://github.com//rapidsai/raft/pull/168)) [@viclafargue](https://github.com/viclafargue)
- Updating sparse prims based on recent changes ([#166](https://github.com//rapidsai/raft/pull/166)) [@cjnolet](https://github.com/cjnolet)
- Prepare Changelog for Automation ([#164](https://github.com//rapidsai/raft/pull/164)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update 0.18 changelog entry ([#163](https://github.com//rapidsai/raft/pull/163)) [@ajschmidt8](https://github.com/ajschmidt8)
- MST symmetric/non-symmetric output for SLHC ([#162](https://github.com//rapidsai/raft/pull/162)) [@divyegala](https://github.com/divyegala)
- Pass pre-computed colors to MST ([#154](https://github.com//rapidsai/raft/pull/154)) [@divyegala](https://github.com/divyegala)
- Streams upgrade in RAFT handle (RMM backend + create handle from parent&#39;s pool) ([#148](https://github.com//rapidsai/raft/pull/148)) [@afender](https://github.com/afender)
- Merge branch-0.18 into 0.19 ([#146](https://github.com//rapidsai/raft/pull/146)) [@dantegd](https://github.com/dantegd)
- Add device_send, device_recv, device_sendrecv, device_multicast_sendrecv ([#144](https://github.com//rapidsai/raft/pull/144)) [@seunghwak](https://github.com/seunghwak)
- Adding SLHC prims. ([#140](https://github.com//rapidsai/raft/pull/140)) [@cjnolet](https://github.com/cjnolet)
- Moving cuml sparse prims to raft ([#139](https://github.com//rapidsai/raft/pull/139)) [@cjnolet](https://github.com/cjnolet)

# raft 0.18.0 (24 Feb 2021)

## Breaking Changes 🚨

- Make NCCL root initialization configurable. (#120) @drobison00

## Bug Fixes 🐛

- Add idx_t template parameter to matrix helper routines (#131) @tfeher
- Eliminate CUDA 10.2 as valid for large svd solving (#129) @wphicks
- Update check to allow svd solver on CUDA&gt;=10.2 (#125) @wphicks
- Updating gpu build.sh and debugging threads CI issue (#123) @dantegd

## New Features 🚀

- Adding additional distances (#116) @cjnolet

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#152) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#150) @Ethyling
- Prepare Changelog for Automation (#135) @ajschmidt8
- Adding Jensen-Shannon and BrayCurtis to DistanceType for Nearest Neighbors (#132) @lowener
- Add brute force KNN (#126) @hlinsen
- Make NCCL root initialization configurable. (#120) @drobison00
- Auto-label PRs based on their content (#117) @jolorunyomi
- Add gather &amp; gatherv to raft::comms::comms_t (#114) @seunghwak
- Adding canberra and chebyshev to distance types (#99) @cjnolet
- Gpuciscripts clean and update (#92) @msadang

# RAFT 0.17.0 (10 Dec 2020)

## New Features
- PR #65: Adding cuml prims that break circular dependency between cuml and cumlprims projects
- PR #101: MST core solver
- PR #93: Incorporate Date/Nagi implementation of Hungarian Algorithm
- PR #94: Allow generic reductions for the map then reduce op
- PR #95: Cholesky rank one update prim

## Improvements
- PR #108: Remove unused old-gpubuild.sh
- PR #73: Move DistanceType enum from cuML to RAFT
- pr #92: Cleanup gpuCI scripts
- PR #98: Adding InnerProduct to DistanceType
- PR #103: Epsilon parameter for Cholesky rank one update
- PR #100: Add divyegala as codeowner
- PR #111: Cleanup gpuCI scripts
- PR #120: Update NCCL init process to support root node placement.

## Bug Fixes
- PR #106: Specify dependency branches to avoid pip resolver failure
- PR #77: Fixing CUB include for CUDA < 11
- PR #86: Missing headers for newly moved prims
- PR #102: Check alignment before binaryOp dispatch
- PR #104: Fix update-version.sh
- PR #109: Fixing Incorrect Deallocation Size and Count Bugs

# RAFT 0.16.0 (Date TBD)

## New Features

- PR #63: Adding MPI comms implementation
- PR #70: Adding CUB to RAFT cmake

## Improvements
- PR #59: Adding csrgemm2 to cusparse_wrappers.h
- PR #61: Add cusparsecsr2dense to cusparse_wrappers.h
- PR #62: Adding `get_device_allocator` to `handle.pxd`
- PR #67: Remove dependence on run-time type info

## Bug Fixes
- PR #56: Fix compiler warnings.
- PR #64: Remove `cublas_try` from `cusolver_wrappers.h`
- PR #66: Fixing typo `get_stream` to `getStream` in `handle.pyx`
- PR #68: Change the type of recvcounts & displs in allgatherv from size_t[] to size_t* and int[] to size_t*, respectively.
- PR #69: Updates for RMM being header only
- PR #74: Fix std_comms::comm_split bug
- PR #79: remove debug print statements
- PR #81: temporarily expose internal NCCL communicator

# RAFT 0.15.0 (Date TBD)

## New Features
- PR #12: Spectral clustering.
- PR #7: Migrating cuml comms -> raft comms_t
- PR #18: Adding commsplit to cuml communicator
- PR #15: add exception based error handling macros
- PR #29: Add ceildiv functionality
- PR #44: Add get_subcomm and set_subcomm to handle_t

## Improvements
- PR #13: Add RMM_INCLUDE and RMM_LIBRARY options to allow linking to non-conda RMM
- PR #22: Preserve order in comms workers for rank initialization
- PR #38: Remove #include <cudar_utils.h> from `raft/mr/`
- PR #39: Adding a virtual destructor to `raft::handle_t` and `raft::comms::comms_t`
- PR #37: Clean-up CUDA related utilities
- PR #41: Upgrade to `cusparseSpMV()`, alg selection, and rectangular matrices.
- PR #45: Add Ampere target to cuda11 cmake
- PR #47: Use gtest conda package in CMake/build.sh by default

## Bug Fixes
- PR #17: Make destructor inline to avoid redeclaration error
- PR #25: Fix bug in handle_t::get_internal_streams
- PR #26: Fix bug in RAFT_EXPECTS (add parentheses surrounding cond)
- PR #34: Fix issue with incorrect docker image being used in local build script
- PR #35: Remove #include <nccl.h> from `raft/error.hpp`
- PR #40: Preemptively fixed future CUDA 11 related errors.
- PR #43: Fixed CUDA version selection mechanism for SpMV.
- PR #46: Fix for cpp file extension issue (nvcc-enforced).
- PR #48: Fix gtest target names in cmake build gtest option.
- PR #49: Skip raft comms test if raft module doesn't exist

# RAFT 0.14.0 (Date TBD)

## New Features
- Initial RAFT version
- PR #3: defining raft::handle_t, device_buffer, host_buffer, allocator classes

## Bug Fixes
- PR #5: Small build.sh fixes
