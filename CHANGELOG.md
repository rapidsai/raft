# raft 22.10.00 (Date TBD)

Please see https://github.com/rapidsai/raft/releases/tag/v22.10.00a for the latest changes to this development branch.

# raft 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Update `mdspan` to account for changes to `extents` ([#751](https://github.com/rapidsai/raft/pull/751)) [@divyegala](https://github.com/divyegala)
- Replace csr_adj_graph functions with faster equivalent ([#746](https://github.com/rapidsai/raft/pull/746)) [@ahendriksen](https://github.com/ahendriksen)
- Integrate KNN implementation: ivf-flat ([#652](https://github.com/rapidsai/raft/pull/652)) [@achirkin](https://github.com/achirkin)
- Moving kmeans from cuml to Raft ([#605](https://github.com/rapidsai/raft/pull/605)) [@lowener](https://github.com/lowener)

## 🐛 Bug Fixes

- Relax ivf-flat test recall thresholds ([#766](https://github.com/rapidsai/raft/pull/766)) [@achirkin](https://github.com/achirkin)
- Restrict the use of `]` to CXX 20 only. ([#764](https://github.com/rapidsai/raft/pull/764)) [@trivialfis](https://github.com/trivialfis)
- Update rapids-cmake version for pyraft in update-version.sh ([#749](https://github.com/rapidsai/raft/pull/749)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Use documented header template for doxygen ([#773](https://github.com/rapidsai/raft/pull/773)) [@galipremsagar](https://github.com/galipremsagar)
- Switch `language` from `None` to `&quot;en&quot;` in docs build ([#721](https://github.com/rapidsai/raft/pull/721)) [@galipremsagar](https://github.com/galipremsagar)

## 🚀 New Features

- Update `mdspan` to account for changes to `extents` ([#751](https://github.com/rapidsai/raft/pull/751)) [@divyegala](https://github.com/divyegala)
- Implement matrix transpose with mdspan. ([#739](https://github.com/rapidsai/raft/pull/739)) [@trivialfis](https://github.com/trivialfis)
- Implement unravel_index for row-major array. ([#723](https://github.com/rapidsai/raft/pull/723)) [@trivialfis](https://github.com/trivialfis)
- Integrate KNN implementation: ivf-flat ([#652](https://github.com/rapidsai/raft/pull/652)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Use common `js` and `css` code ([#779](https://github.com/rapidsai/raft/pull/779)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` &amp; `distributed` for release ([#772](https://github.com/rapidsai/raft/pull/772)) [@galipremsagar](https://github.com/galipremsagar)
- Move cmake to the build section. ([#763](https://github.com/rapidsai/raft/pull/763)) [@vyasr](https://github.com/vyasr)
- Adding old kmeans impl back in (as kmeans_deprecated) ([#761](https://github.com/rapidsai/raft/pull/761)) [@cjnolet](https://github.com/cjnolet)
- Fix for KMeans raw pointers API ([#758](https://github.com/rapidsai/raft/pull/758)) [@lowener](https://github.com/lowener)
- Fix KMeans ([#756](https://github.com/rapidsai/raft/pull/756)) [@divyegala](https://github.com/divyegala)
- Add inline to nccl_sync_stream() ([#750](https://github.com/rapidsai/raft/pull/750)) [@seunghwak](https://github.com/seunghwak)
- Replace csr_adj_graph functions with faster equivalent ([#746](https://github.com/rapidsai/raft/pull/746)) [@ahendriksen](https://github.com/ahendriksen)
- Add wrapper functions for ncclGroupStart() and ncclGroupEnd() ([#742](https://github.com/rapidsai/raft/pull/742)) [@seunghwak](https://github.com/seunghwak)
- Fix variadic template type check for mdarrays ([#741](https://github.com/rapidsai/raft/pull/741)) [@hlinsen](https://github.com/hlinsen)
- RMAT rectangular graph generator ([#738](https://github.com/rapidsai/raft/pull/738)) [@teju85](https://github.com/teju85)
- Update conda recipes to UCX 1.13.0 ([#736](https://github.com/rapidsai/raft/pull/736)) [@pentschev](https://github.com/pentschev)
- Add warp-aggregated atomic increment ([#735](https://github.com/rapidsai/raft/pull/735)) [@ahendriksen](https://github.com/ahendriksen)
- fix logic bug in include_checker.py utility ([#734](https://github.com/rapidsai/raft/pull/734)) [@grlee77](https://github.com/grlee77)
- Support 32bit and unsigned indices in bruteforce KNN ([#730](https://github.com/rapidsai/raft/pull/730)) [@achirkin](https://github.com/achirkin)
- Ability to use ccache to speedup local builds ([#729](https://github.com/rapidsai/raft/pull/729)) [@teju85](https://github.com/teju85)
- Pin max version of `cuda-python` to `11.7.0` ([#728](https://github.com/rapidsai/raft/pull/728)) [@Ethyling](https://github.com/Ethyling)
- Always add `raft::raft_nn_lib` and `raft::raft_distance_lib` aliases ([#727](https://github.com/rapidsai/raft/pull/727)) [@trxcllnt](https://github.com/trxcllnt)
- Add several type aliases and helpers for creating mdarrays ([#726](https://github.com/rapidsai/raft/pull/726)) [@achirkin](https://github.com/achirkin)
- fix nans in naive kl divergence kernel introduced by div by 0. ([#724](https://github.com/rapidsai/raft/pull/724)) [@mdoijade](https://github.com/mdoijade)
- Use rapids-cmake for cuco ([#722](https://github.com/rapidsai/raft/pull/722)) [@vyasr](https://github.com/vyasr)
- Update Python classifiers. ([#719](https://github.com/rapidsai/raft/pull/719)) [@bdice](https://github.com/bdice)
- Fix sccache ([#718](https://github.com/rapidsai/raft/pull/718)) [@Ethyling](https://github.com/Ethyling)
- Introducing raft::mdspan as an alias ([#715](https://github.com/rapidsai/raft/pull/715)) [@divyegala](https://github.com/divyegala)
- Update cuco version ([#714](https://github.com/rapidsai/raft/pull/714)) [@vyasr](https://github.com/vyasr)
- Update conda environment pinnings and update-versions.sh. ([#713](https://github.com/rapidsai/raft/pull/713)) [@bdice](https://github.com/bdice)
- Branch 22.08 merge branch 22.06 ([#712](https://github.com/rapidsai/raft/pull/712)) [@cjnolet](https://github.com/cjnolet)
- Testing conda compilers ([#705](https://github.com/rapidsai/raft/pull/705)) [@cjnolet](https://github.com/cjnolet)
- Unpin `dask` &amp; `distributed` for development ([#704](https://github.com/rapidsai/raft/pull/704)) [@galipremsagar](https://github.com/galipremsagar)
- Avoid shadowing CMAKE_ARGS variable in build.sh ([#701](https://github.com/rapidsai/raft/pull/701)) [@vyasr](https://github.com/vyasr)
- Use unique ptr in `print_device_vector` ([#695](https://github.com/rapidsai/raft/pull/695)) [@lowener](https://github.com/lowener)
- Add missing Thrust includes ([#678](https://github.com/rapidsai/raft/pull/678)) [@bdice](https://github.com/bdice)
- Consolidate C++ conda recipes and add libraft-tests package ([#641](https://github.com/rapidsai/raft/pull/641)) [@Ethyling](https://github.com/Ethyling)
- Moving kmeans from cuml to Raft ([#605](https://github.com/rapidsai/raft/pull/605)) [@lowener](https://github.com/lowener)

# raft 22.06.00 (7 Jun 2022)

## 🚨 Breaking Changes

- Rng: removed cyclic dependency creating hard-to-debug compiler errors ([#639](https://github.com/rapidsai/raft/pull/639)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Allow enabling NVTX markers by downstream projects after install ([#610](https://github.com/rapidsai/raft/pull/610)) [@achirkin](https://github.com/achirkin)
- Rng: expose host-rng-state in host-only API ([#609](https://github.com/rapidsai/raft/pull/609)) [@MatthiasKohl](https://github.com/MatthiasKohl)

## 🐛 Bug Fixes

- For fixing the cuGraph test failures with PCG ([#690](https://github.com/rapidsai/raft/pull/690)) [@vinaydes](https://github.com/vinaydes)
- Fix excessive memory used in selection test ([#689](https://github.com/rapidsai/raft/pull/689)) [@achirkin](https://github.com/achirkin)
- Revert print vector changes because of std::vector&lt;bool&gt; ([#681](https://github.com/rapidsai/raft/pull/681)) [@lowener](https://github.com/lowener)
- fix race in fusedL2knn smem read/write by adding a syncwarp ([#679](https://github.com/rapidsai/raft/pull/679)) [@mdoijade](https://github.com/mdoijade)
- gemm: fix  parameter C mistakenly set as const ([#664](https://github.com/rapidsai/raft/pull/664)) [@achirkin](https://github.com/achirkin)
- Fix SelectionTest: allow different indices when keys are equal. ([#659](https://github.com/rapidsai/raft/pull/659)) [@achirkin](https://github.com/achirkin)
- Revert recent cmake updates ([#657](https://github.com/rapidsai/raft/pull/657)) [@cjnolet](https://github.com/cjnolet)
- Don&#39;t install component dependency files in raft-header only mode ([#655](https://github.com/rapidsai/raft/pull/655)) [@robertmaynard](https://github.com/robertmaynard)
- Rng: removed cyclic dependency creating hard-to-debug compiler errors ([#639](https://github.com/rapidsai/raft/pull/639)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Fixing raft compile bug w/ RNG changes ([#634](https://github.com/rapidsai/raft/pull/634)) [@cjnolet](https://github.com/cjnolet)
- Get `libcudacxx` from `cuco` ([#632](https://github.com/rapidsai/raft/pull/632)) [@trxcllnt](https://github.com/trxcllnt)
- RNG API fixes ([#630](https://github.com/rapidsai/raft/pull/630)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Fix mdspan accessor mixin offset policy. ([#628](https://github.com/rapidsai/raft/pull/628)) [@trivialfis](https://github.com/trivialfis)
- Branch 22.06 merge 22.04 ([#625](https://github.com/rapidsai/raft/pull/625)) [@cjnolet](https://github.com/cjnolet)
- fix issue in fusedL2knn which happens when rows are multiple of 256 ([#604](https://github.com/rapidsai/raft/pull/604)) [@mdoijade](https://github.com/mdoijade)

## 🚀 New Features

- Restore changes from #653 and #655 and correct cmake component dependencies ([#686](https://github.com/rapidsai/raft/pull/686)) [@robertmaynard](https://github.com/robertmaynard)
- Adding handle and stream to pylibraft ([#683](https://github.com/rapidsai/raft/pull/683)) [@cjnolet](https://github.com/cjnolet)
- Map CMake install components to conda library packages ([#653](https://github.com/rapidsai/raft/pull/653)) [@robertmaynard](https://github.com/robertmaynard)
- Rng: expose host-rng-state in host-only API ([#609](https://github.com/rapidsai/raft/pull/609)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- mdspan/mdarray template functions and utilities ([#601](https://github.com/rapidsai/raft/pull/601)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Change build.sh to find C++ library by default ([#697](https://github.com/rapidsai/raft/pull/697)) [@vyasr](https://github.com/vyasr)
- Pin `dask` and `distributed` for release ([#693](https://github.com/rapidsai/raft/pull/693)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` &amp; `distributed` for release ([#680](https://github.com/rapidsai/raft/pull/680)) [@galipremsagar](https://github.com/galipremsagar)
- Improve logging ([#673](https://github.com/rapidsai/raft/pull/673)) [@achirkin](https://github.com/achirkin)
- Fix minor errors in CMake configuration ([#662](https://github.com/rapidsai/raft/pull/662)) [@vyasr](https://github.com/vyasr)
- Pulling mdspan fork (from official rapids repo) into raft to remove dependency ([#649](https://github.com/rapidsai/raft/pull/649)) [@cjnolet](https://github.com/cjnolet)
- Fixing the unit test issue(s) in RAFT ([#646](https://github.com/rapidsai/raft/pull/646)) [@vinaydes](https://github.com/vinaydes)
- Build pyraft with scikit-build ([#644](https://github.com/rapidsai/raft/pull/644)) [@vyasr](https://github.com/vyasr)
- Some fixes to pairwise distances for cupy integration ([#643](https://github.com/rapidsai/raft/pull/643)) [@cjnolet](https://github.com/cjnolet)
- Require UCX 1.12.1+ ([#638](https://github.com/rapidsai/raft/pull/638)) [@jakirkham](https://github.com/jakirkham)
- Updating raft rng host public API and adding docs ([#636](https://github.com/rapidsai/raft/pull/636)) [@cjnolet](https://github.com/cjnolet)
- Build pylibraft with scikit-build ([#633](https://github.com/rapidsai/raft/pull/633)) [@vyasr](https://github.com/vyasr)
- Add `cuda_lib_dir` to `library_dirs`, allow changing `UCX`/`RMM`/`Thrust`/`spdlog` locations via envvars in `setup.py` ([#624](https://github.com/rapidsai/raft/pull/624)) [@trxcllnt](https://github.com/trxcllnt)
- Remove perf prints from MST ([#623](https://github.com/rapidsai/raft/pull/623)) [@divyegala](https://github.com/divyegala)
- Enable components installation using CMake ([#621](https://github.com/rapidsai/raft/pull/621)) [@Ethyling](https://github.com/Ethyling)
- Allow nullptr as input-indices argument of select_k ([#618](https://github.com/rapidsai/raft/pull/618)) [@achirkin](https://github.com/achirkin)
- Update CMake pinning to allow newer CMake versions ([#617](https://github.com/rapidsai/raft/pull/617)) [@vyasr](https://github.com/vyasr)
- Unpin `dask` &amp; `distributed` for development ([#616](https://github.com/rapidsai/raft/pull/616)) [@galipremsagar](https://github.com/galipremsagar)
- Improve performance of select-top-k RADIX implementation ([#615](https://github.com/rapidsai/raft/pull/615)) [@achirkin](https://github.com/achirkin)
- Moving more prims benchmarks to RAFT ([#613](https://github.com/rapidsai/raft/pull/613)) [@cjnolet](https://github.com/cjnolet)
- Allow enabling NVTX markers by downstream projects after install ([#610](https://github.com/rapidsai/raft/pull/610)) [@achirkin](https://github.com/achirkin)
- Improve performance of select-top-k  WARP_SORT implementation ([#606](https://github.com/rapidsai/raft/pull/606)) [@achirkin](https://github.com/achirkin)
- Enable building static libs ([#602](https://github.com/rapidsai/raft/pull/602)) [@trxcllnt](https://github.com/trxcllnt)
- Update `ucx-py` version ([#596](https://github.com/rapidsai/raft/pull/596)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix merge conflicts ([#587](https://github.com/rapidsai/raft/pull/587)) [@ajschmidt8](https://github.com/ajschmidt8)
- Making cuco, thrust, and mdspan optional dependencies. ([#585](https://github.com/rapidsai/raft/pull/585)) [@cjnolet](https://github.com/cjnolet)
- Some RBC3D fixes ([#530](https://github.com/rapidsai/raft/pull/530)) [@cjnolet](https://github.com/cjnolet)

# raft 22.04.00 (6 Apr 2022)

## 🚨 Breaking Changes

- Moving some of the remaining linalg prims from cuml ([#502](https://github.com/rapidsai/raft/pull/502)) [@cjnolet](https://github.com/cjnolet)
- Fix badly merged cublas wrappers ([#492](https://github.com/rapidsai/raft/pull/492)) [@achirkin](https://github.com/achirkin)
- Hiding implementation details for lap, clustering, spectral, and label ([#477](https://github.com/rapidsai/raft/pull/477)) [@cjnolet](https://github.com/cjnolet)
- Adding destructor for std comms and using nccl allreduce for barrier in mpi comms ([#473](https://github.com/rapidsai/raft/pull/473)) [@cjnolet](https://github.com/cjnolet)
- Cleaning up cusparse_wrappers ([#441](https://github.com/rapidsai/raft/pull/441)) [@cjnolet](https://github.com/cjnolet)
- Improvents to RNG ([#434](https://github.com/rapidsai/raft/pull/434)) [@vinaydes](https://github.com/vinaydes)
- Remove RAFT memory management ([#400](https://github.com/rapidsai/raft/pull/400)) [@viclafargue](https://github.com/viclafargue)
- LinAlg impl in detail ([#383](https://github.com/rapidsai/raft/pull/383)) [@divyegala](https://github.com/divyegala)

## 🐛 Bug Fixes

- Pin cmake in conda recipe to &lt;3.23 ([#600](https://github.com/rapidsai/raft/pull/600)) [@dantegd](https://github.com/dantegd)
- Fix make_device_vector_view ([#595](https://github.com/rapidsai/raft/pull/595)) [@lowener](https://github.com/lowener)
- Update cuco version. ([#592](https://github.com/rapidsai/raft/pull/592)) [@vyasr](https://github.com/vyasr)
- Fixing raft headers dir ([#574](https://github.com/rapidsai/raft/pull/574)) [@cjnolet](https://github.com/cjnolet)
- Update update-version.sh ([#560](https://github.com/rapidsai/raft/pull/560)) [@raydouglass](https://github.com/raydouglass)
- find_package(raft) can now be called multiple times safely ([#532](https://github.com/rapidsai/raft/pull/532)) [@robertmaynard](https://github.com/robertmaynard)
- Allocate sufficient memory for Hungarian if number of batches &gt; 1 ([#531](https://github.com/rapidsai/raft/pull/531)) [@ChuckHastings](https://github.com/ChuckHastings)
- Adding lap.hpp back (with deprecation) ([#529](https://github.com/rapidsai/raft/pull/529)) [@cjnolet](https://github.com/cjnolet)
- raft-config is idempotent no matter RAFT_COMPILE_LIBRARIES value ([#516](https://github.com/rapidsai/raft/pull/516)) [@robertmaynard](https://github.com/robertmaynard)
- Call initialize() in mpi_comms_t constructor. ([#506](https://github.com/rapidsai/raft/pull/506)) [@seunghwak](https://github.com/seunghwak)
- Improve row-major meanvar kernel via minimizing atomicCAS locks ([#489](https://github.com/rapidsai/raft/pull/489)) [@achirkin](https://github.com/achirkin)
- Adding destructor for std comms and using nccl allreduce for barrier in mpi comms ([#473](https://github.com/rapidsai/raft/pull/473)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- Updating docs for 22.04 ([#566](https://github.com/rapidsai/raft/pull/566)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add benchmarks ([#549](https://github.com/rapidsai/raft/pull/549)) [@achirkin](https://github.com/achirkin)
- Unify weighted mean code ([#514](https://github.com/rapidsai/raft/pull/514)) [@lowener](https://github.com/lowener)
- single-pass raft::stats::meanvar ([#472](https://github.com/rapidsai/raft/pull/472)) [@achirkin](https://github.com/achirkin)
- Move `random` package of cuML to RAFT ([#449](https://github.com/rapidsai/raft/pull/449)) [@divyegala](https://github.com/divyegala)
- mdspan integration. ([#437](https://github.com/rapidsai/raft/pull/437)) [@trivialfis](https://github.com/trivialfis)
- Interruptible execution ([#433](https://github.com/rapidsai/raft/pull/433)) [@achirkin](https://github.com/achirkin)
- make raft sources compilable with clang ([#424](https://github.com/rapidsai/raft/pull/424)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Span implementation. ([#399](https://github.com/rapidsai/raft/pull/399)) [@trivialfis](https://github.com/trivialfis)

## 🛠️ Improvements

- Adding build script for docs ([#589](https://github.com/rapidsai/raft/pull/589)) [@cjnolet](https://github.com/cjnolet)
- Temporarily disable new `ops-bot` functionality ([#586](https://github.com/rapidsai/raft/pull/586)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix commands to get conda output files ([#584](https://github.com/rapidsai/raft/pull/584)) [@Ethyling](https://github.com/Ethyling)
- Link to `cuco` and add faiss `EXCLUDE_FROM_ALL` option ([#583](https://github.com/rapidsai/raft/pull/583)) [@trxcllnt](https://github.com/trxcllnt)
- exposing faiss::faiss ([#582](https://github.com/rapidsai/raft/pull/582)) [@cjnolet](https://github.com/cjnolet)
- Pin `dask` and `distributed` version ([#581](https://github.com/rapidsai/raft/pull/581)) [@galipremsagar](https://github.com/galipremsagar)
- removing exclude_from_all from cuco ([#580](https://github.com/rapidsai/raft/pull/580)) [@cjnolet](https://github.com/cjnolet)
- Adding INSTALL_EXPORT_SET for cuco, rmm, thrust ([#579](https://github.com/rapidsai/raft/pull/579)) [@cjnolet](https://github.com/cjnolet)
- Thrust package name case ([#576](https://github.com/rapidsai/raft/pull/576)) [@trxcllnt](https://github.com/trxcllnt)
- Add missing thrust includes to transpose.cuh ([#575](https://github.com/rapidsai/raft/pull/575)) [@zbjornson](https://github.com/zbjornson)
- Use unanchored clang-format version check ([#573](https://github.com/rapidsai/raft/pull/573)) [@zbjornson](https://github.com/zbjornson)
- Fixing accidental removal of thrust target from cmakelists ([#571](https://github.com/rapidsai/raft/pull/571)) [@cjnolet](https://github.com/cjnolet)
- Don&#39;t add gtest to build export set or generate a gtest-config.cmake ([#565](https://github.com/rapidsai/raft/pull/565)) [@trxcllnt](https://github.com/trxcllnt)
- Set `main` label by default ([#559](https://github.com/rapidsai/raft/pull/559)) [@galipremsagar](https://github.com/galipremsagar)
- Add local conda channel while looking for conda outputs ([#558](https://github.com/rapidsai/raft/pull/558)) [@Ethyling](https://github.com/Ethyling)
- Updated dask and distributed to &gt;=2022.02.1 ([#557](https://github.com/rapidsai/raft/pull/557)) [@rlratzel](https://github.com/rlratzel)
- Upload packages using testing label for nightlies ([#556](https://github.com/rapidsai/raft/pull/556)) [@Ethyling](https://github.com/Ethyling)
- Add `.github/ops-bot.yaml` config file ([#554](https://github.com/rapidsai/raft/pull/554)) [@ajschmidt8](https://github.com/ajschmidt8)
- Disabling benchmarks building by default. ([#553](https://github.com/rapidsai/raft/pull/553)) [@cjnolet](https://github.com/cjnolet)
- KNN select-top-k variants ([#551](https://github.com/rapidsai/raft/pull/551)) [@achirkin](https://github.com/achirkin)
- Adding logger ([#550](https://github.com/rapidsai/raft/pull/550)) [@cjnolet](https://github.com/cjnolet)
- clang-tidy support: improved clang run scripts with latest changes (see cugraph-ops) ([#548](https://github.com/rapidsai/raft/pull/548)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- Pylibraft for pairwise distances ([#540](https://github.com/rapidsai/raft/pull/540)) [@cjnolet](https://github.com/cjnolet)
- mdspan PoC for distance make_blobs ([#538](https://github.com/rapidsai/raft/pull/538)) [@cjnolet](https://github.com/cjnolet)
- Include thrust/sort.h in ball_cover.cuh ([#526](https://github.com/rapidsai/raft/pull/526)) [@akifcorduk](https://github.com/akifcorduk)
- Increase parallelism in allgatherv ([#525](https://github.com/rapidsai/raft/pull/525)) [@seunghwak](https://github.com/seunghwak)
- Moving device functions to cuh files and deprecating hpp ([#524](https://github.com/rapidsai/raft/pull/524)) [@cjnolet](https://github.com/cjnolet)
- Use `dynamic_extent` from `stdex`. ([#523](https://github.com/rapidsai/raft/pull/523)) [@trivialfis](https://github.com/trivialfis)
- Updating some of the ci check scripts ([#522](https://github.com/rapidsai/raft/pull/522)) [@cjnolet](https://github.com/cjnolet)
- Use shfl_xor in warpReduce for broadcast ([#521](https://github.com/rapidsai/raft/pull/521)) [@akifcorduk](https://github.com/akifcorduk)
- Fixing Python conda package and installation ([#520](https://github.com/rapidsai/raft/pull/520)) [@cjnolet](https://github.com/cjnolet)
- Adding instructions to install from conda and build using CPM ([#519](https://github.com/rapidsai/raft/pull/519)) [@cjnolet](https://github.com/cjnolet)
- Implement span storage optimization. ([#515](https://github.com/rapidsai/raft/pull/515)) [@trivialfis](https://github.com/trivialfis)
- RNG test fixes and improvements ([#513](https://github.com/rapidsai/raft/pull/513)) [@vinaydes](https://github.com/vinaydes)
- Moving scores and metrics over to raft::stats ([#512](https://github.com/rapidsai/raft/pull/512)) [@cjnolet](https://github.com/cjnolet)
- Random ball cover in 3d ([#510](https://github.com/rapidsai/raft/pull/510)) [@cjnolet](https://github.com/cjnolet)
- Initializing memory in RBC ([#509](https://github.com/rapidsai/raft/pull/509)) [@cjnolet](https://github.com/cjnolet)
- Adjusting conda packaging to remove duplicate dependencies ([#508](https://github.com/rapidsai/raft/pull/508)) [@cjnolet](https://github.com/cjnolet)
- Moving remaining stats prims from cuml ([#507](https://github.com/rapidsai/raft/pull/507)) [@cjnolet](https://github.com/cjnolet)
- Correcting the namespace ([#505](https://github.com/rapidsai/raft/pull/505)) [@vinaydes](https://github.com/vinaydes)
- Passing stream through commsplit ([#503](https://github.com/rapidsai/raft/pull/503)) [@cjnolet](https://github.com/cjnolet)
- Moving some of the remaining linalg prims from cuml ([#502](https://github.com/rapidsai/raft/pull/502)) [@cjnolet](https://github.com/cjnolet)
- Fixing spectral APIs ([#496](https://github.com/rapidsai/raft/pull/496)) [@cjnolet](https://github.com/cjnolet)
- Fix badly merged cublas wrappers ([#492](https://github.com/rapidsai/raft/pull/492)) [@achirkin](https://github.com/achirkin)
- Fix integer overflow in distances ([#490](https://github.com/rapidsai/raft/pull/490)) [@RAMitchell](https://github.com/RAMitchell)
- Reusing shared libs in gpu ci builds ([#487](https://github.com/rapidsai/raft/pull/487)) [@cjnolet](https://github.com/cjnolet)
- Adding fatbin to shared libs and fixing conda paths in cpu build ([#485](https://github.com/rapidsai/raft/pull/485)) [@cjnolet](https://github.com/cjnolet)
- Add CMake `install` rule for tests ([#483](https://github.com/rapidsai/raft/pull/483)) [@ajschmidt8](https://github.com/ajschmidt8)
- Adding cpu ci for conda build ([#482](https://github.com/rapidsai/raft/pull/482)) [@cjnolet](https://github.com/cjnolet)
- iUpdating codeowners to use new raft codeowners ([#480](https://github.com/rapidsai/raft/pull/480)) [@cjnolet](https://github.com/cjnolet)
- Hiding implementation details for lap, clustering, spectral, and label ([#477](https://github.com/rapidsai/raft/pull/477)) [@cjnolet](https://github.com/cjnolet)
- Define PTDS via `-D` to fix cache misses in sccache ([#476](https://github.com/rapidsai/raft/pull/476)) [@trxcllnt](https://github.com/trxcllnt)
- Unpin dask and distributed ([#474](https://github.com/rapidsai/raft/pull/474)) [@galipremsagar](https://github.com/galipremsagar)
- Replace `ccache` with `sccache` ([#471](https://github.com/rapidsai/raft/pull/471)) [@ajschmidt8](https://github.com/ajschmidt8)
- More README updates ([#467](https://github.com/rapidsai/raft/pull/467)) [@cjnolet](https://github.com/cjnolet)
- CUBLAS wrappers with switchable host/device pointer mode ([#453](https://github.com/rapidsai/raft/pull/453)) [@achirkin](https://github.com/achirkin)
- Cleaning up cusparse_wrappers ([#441](https://github.com/rapidsai/raft/pull/441)) [@cjnolet](https://github.com/cjnolet)
- Adding conda packaging for libraft and pyraft ([#439](https://github.com/rapidsai/raft/pull/439)) [@cjnolet](https://github.com/cjnolet)
- Improvents to RNG ([#434](https://github.com/rapidsai/raft/pull/434)) [@vinaydes](https://github.com/vinaydes)
- Hiding implementation details for comms ([#409](https://github.com/rapidsai/raft/pull/409)) [@cjnolet](https://github.com/cjnolet)
- Remove RAFT memory management ([#400](https://github.com/rapidsai/raft/pull/400)) [@viclafargue](https://github.com/viclafargue)
- LinAlg impl in detail ([#383](https://github.com/rapidsai/raft/pull/383)) [@divyegala](https://github.com/divyegala)

# raft 22.02.00 (2 Feb 2022)

## 🚨 Breaking Changes

- Simplify raft component CMake logic, and allow compilation without FAISS ([#428](https://github.com/rapidsai/raft/pull/428)) [@robertmaynard](https://github.com/robertmaynard)
- One cudaStream_t instance per raft::handle_t ([#291](https://github.com/rapidsai/raft/pull/291)) [@divyegala](https://github.com/divyegala)

## 🐛 Bug Fixes

- Removing extra logging from faiss mr ([#463](https://github.com/rapidsai/raft/pull/463)) [@cjnolet](https://github.com/cjnolet)
- Pin `dask` &amp; `distributed` versions ([#455](https://github.com/rapidsai/raft/pull/455)) [@galipremsagar](https://github.com/galipremsagar)
- Replace RMM CUDA Python bindings with those provided  by CUDA-Python ([#451](https://github.com/rapidsai/raft/pull/451)) [@shwina](https://github.com/shwina)
- Fix comms memory leak ([#436](https://github.com/rapidsai/raft/pull/436)) [@seunghwak](https://github.com/seunghwak)
- Fix C++ doxygen documentation ([#426](https://github.com/rapidsai/raft/pull/426)) [@achirkin](https://github.com/achirkin)
- Fix clang-format style errors ([#425](https://github.com/rapidsai/raft/pull/425)) [@achirkin](https://github.com/achirkin)
- Fix using incorrect macro RAFT_CHECK_CUDA in place of RAFT_CUDA_TRY ([#415](https://github.com/rapidsai/raft/pull/415)) [@achirkin](https://github.com/achirkin)
- Fix CUDA_CHECK_NO_THROW compatibility define ([#414](https://github.com/rapidsai/raft/pull/414)) [@zbjornson](https://github.com/zbjornson)
- Disabling fused l2 knn from bfknn ([#407](https://github.com/rapidsai/raft/pull/407)) [@cjnolet](https://github.com/cjnolet)
- Disabling expanded fused l2 knn to unblock cuml CI ([#404](https://github.com/rapidsai/raft/pull/404)) [@cjnolet](https://github.com/cjnolet)
- Reverting default knn distance to L2Unexpanded for now. ([#403](https://github.com/rapidsai/raft/pull/403)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- README and build fixes before release ([#459](https://github.com/rapidsai/raft/pull/459)) [@cjnolet](https://github.com/cjnolet)
- Updates to Python and C++ Docs ([#442](https://github.com/rapidsai/raft/pull/442)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- error macros: determining buffer size instead of fixed 2048 chars ([#420](https://github.com/rapidsai/raft/pull/420)) [@MatthiasKohl](https://github.com/MatthiasKohl)
- NVTX range helpers ([#416](https://github.com/rapidsai/raft/pull/416)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Splitting fused l2 knn specializations ([#461](https://github.com/rapidsai/raft/pull/461)) [@cjnolet](https://github.com/cjnolet)
- Update cuCollection git tag ([#447](https://github.com/rapidsai/raft/pull/447)) [@seunghwak](https://github.com/seunghwak)
- Remove libcudacxx patch needed for nvcc 11.4 ([#446](https://github.com/rapidsai/raft/pull/446)) [@robertmaynard](https://github.com/robertmaynard)
- Unpin `dask` and `distributed` ([#440](https://github.com/rapidsai/raft/pull/440)) [@galipremsagar](https://github.com/galipremsagar)
- Public apis for remainder of matrix and stats ([#438](https://github.com/rapidsai/raft/pull/438)) [@divyegala](https://github.com/divyegala)
- Fix bug in producer-consumer buffer exchange which occurs in UMAP test on GV100 ([#429](https://github.com/rapidsai/raft/pull/429)) [@mdoijade](https://github.com/mdoijade)
- Simplify raft component CMake logic, and allow compilation without FAISS ([#428](https://github.com/rapidsai/raft/pull/428)) [@robertmaynard](https://github.com/robertmaynard)
- Update ucx-py version on release using rvc ([#422](https://github.com/rapidsai/raft/pull/422)) [@Ethyling](https://github.com/Ethyling)
- Disabling fused l2 knn again. Not sure how this got added back. ([#421](https://github.com/rapidsai/raft/pull/421)) [@cjnolet](https://github.com/cjnolet)
- Adding no throw macro variants ([#417](https://github.com/rapidsai/raft/pull/417)) [@cjnolet](https://github.com/cjnolet)
- Remove `IncludeCategories` from `.clang-format` ([#412](https://github.com/rapidsai/raft/pull/412)) [@codereport](https://github.com/codereport)
- fix nan issues in L2 expanded sqrt KNN distances ([#411](https://github.com/rapidsai/raft/pull/411)) [@mdoijade](https://github.com/mdoijade)
- Consistent renaming of CHECK_CUDA and *_TRY macros ([#410](https://github.com/rapidsai/raft/pull/410)) [@cjnolet](https://github.com/cjnolet)
- Faster matrix-vector-ops ([#401](https://github.com/rapidsai/raft/pull/401)) [@achirkin](https://github.com/achirkin)
- Adding dev conda environment files. ([#397](https://github.com/rapidsai/raft/pull/397)) [@cjnolet](https://github.com/cjnolet)
- Update to UCX-Py 0.24 ([#392](https://github.com/rapidsai/raft/pull/392)) [@pentschev](https://github.com/pentschev)
- Branch 21.12 merge 22.02 ([#386](https://github.com/rapidsai/raft/pull/386)) [@cjnolet](https://github.com/cjnolet)
- Hiding implementation details for sparse API ([#381](https://github.com/rapidsai/raft/pull/381)) [@cjnolet](https://github.com/cjnolet)
- Adding distance specializations ([#376](https://github.com/rapidsai/raft/pull/376)) [@cjnolet](https://github.com/cjnolet)
- Use FAISS with RMM ([#363](https://github.com/rapidsai/raft/pull/363)) [@viclafargue](https://github.com/viclafargue)
- Add Fused L2 Expanded KNN kernel ([#339](https://github.com/rapidsai/raft/pull/339)) [@mdoijade](https://github.com/mdoijade)
- Update `.clang-format` to be consistent with all other RAPIDS repos ([#300](https://github.com/rapidsai/raft/pull/300)) [@codereport](https://github.com/codereport)
- One cudaStream_t instance per raft::handle_t ([#291](https://github.com/rapidsai/raft/pull/291)) [@divyegala](https://github.com/divyegala)

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
