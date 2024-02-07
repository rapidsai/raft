# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import unittest

from raft_ann_bench.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def clear_loggers():
    """Remove handlers from all loggers"""
    import logging

    loggers = [logging.getLogger()] + list(
        logging.Logger.manager.loggerDict.values()
    )
    for logger in loggers:
        handlers = getattr(logger, "handlers", [])
        for handler in handlers:
            logger.removeHandler(handler)


class TestCli(unittest.TestCase):
    def setUp(self):
        # Fix pytest: https://github.com/pytest-dev/pytest/issues/5502
        clear_loggers()

        self.run_algorithms = "raft_cagra"
        self.run_dataset = "sift-128-euclidean"

    def test_main(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_run(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_build(self):
        result = runner.invoke(
            app,
            [
                "run",
                "--algorithms={}".format(self.run_algorithms),
                "--dataset={}".format(self.run_dataset),
                "build",
                "--help",
            ],
        )
        assert result.exit_code == 0

        result = runner.invoke(
            app,
            [
                "run",
                "--dry-run",
                "--algorithms={}".format(self.run_algorithms),
                "--dataset={}".format(self.run_dataset),
                "build",
            ],
        )
        assert result.exit_code == 0

    def test_run_search(self):
        result = runner.invoke(
            app,
            [
                "run",
                "--algorithms={}".format(self.run_algorithms),
                "--dataset={}".format(self.run_dataset),
                "search",
                "--help",
            ],
        )
        assert result.exit_code == 0

        result = runner.invoke(
            app,
            args=[
                "run",
                "--dry-run",
                "--algorithms={}".format(self.run_algorithms),
                "--dataset={}".format(self.run_dataset),
                "search",
            ],
            standalone_mode=False,
        )
        assert result.exit_code == 0
