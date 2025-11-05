# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import raft_dask


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(raft_dask.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(raft_dask.__version__, str)
    assert len(raft_dask.__version__) > 0
