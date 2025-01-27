# Copyright (c) 2024, NVIDIA CORPORATION.

import pylibraft


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(pylibraft.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(pylibraft.__version__, str)
    assert len(pylibraft.__version__) > 0
