#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import colors

def main(input_path):
    input_path = Path(input_path)
    print("-- loading data")
    df = pd.read_csv(input_path)

    print("-- analyzing data")
    # Strip spaces from column names
    df = df.rename(columns=str.strip)
    df["seconds"] = df["metric"] / 1000
    df["file"] = df["source file name"]
    df["phase"] = df["phase name"].str.strip()

    def categorize_time(s):
        if s < 60:
            return "less than a minute"
        else:
            return "more than a minute"

    dfp = (df
           # Remove nvcc driver entries. They don't contain a source file name
           .query("phase!='nvcc (driver)'")
           # Make a pivot table containing files as row, phase (preprocessing,
           # cicc, etc.) as column and the total times as table entries. NOTE:
           # if compiled for multiple archs, the archs will be summed.
           .pivot_table(index="file", values="seconds", columns="phase", aggfunc='sum'))

    dfp_sum = dfp.sum(axis="columns")

    df_fraction = dfp.divide(dfp_sum, axis="index")
    df_fraction["total time"] = dfp_sum
    df_fraction = df_fraction.melt(ignore_index=False, id_vars="total time", var_name="phase", value_name="fraction")

    dfp["total time"] = dfp_sum
    df_absolute = dfp.melt(ignore_index=False, id_vars="total time", var_name="phase", value_name="seconds")

    df_fraction["time category"] = dfp["total time"].apply(categorize_time)
    df_absolute["time category"] = dfp["total time"].apply(categorize_time)

    # host: light red to dark red (preprocessing, cudafe, gcc (compiling))
    # device: ligt green to dark green (preprocessing, cicc, ptxas)
    palette = {
        "gcc (preprocessing 4)": colors.hsv_to_rgb((0, 1, 1)),
        'cudafe++': colors.hsv_to_rgb((0, 1, .75)),
        'gcc (compiling)': colors.hsv_to_rgb((0, 1, .4)),
        "gcc (preprocessing 1)": colors.hsv_to_rgb((.33, 1, 1)),
        'cicc': colors.hsv_to_rgb((.33, 1, 0.75)),
        'ptxas': colors.hsv_to_rgb((.33, 1, 0.4)),
        'fatbinary': "grey",
    }

    print("-- Ten longest translation units:")
    colwidth = pd.get_option('display.max_colwidth') - 1
    dfp = dfp.reset_index()
    dfp["file"] = dfp["file"].apply(lambda s: s[-colwidth:])
    print(dfp.sort_values("total time", ascending=False).reset_index().loc[:10])

    print("-- Plotting absolute compile times")
    abs_out_path = f"{input_path}.absolute.compile_times.png"
    sns.displot(
        df_absolute.sort_values("total time").reset_index(),
        y="file",
        hue="phase",
        hue_order=reversed(
            ["gcc (preprocessing 4)", 'cudafe++', 'gcc (compiling)',
             "gcc (preprocessing 1)", 'cicc', 'ptxas',
             'fatbinary',
        ]),
        palette=palette,
        weights="seconds",
        multiple="stack",
        kind="hist",
        height=20,
    )
    plt.xlabel("seconds");
    plt.savefig(abs_out_path)
    print(f"-- Wrote absolute compile time plot to {abs_out_path}")

    print("-- Plotting relative compile times")
    rel_out_path = f"{input_path}.relative.compile_times.png"
    sns.displot(
        df_fraction.sort_values('total time').reset_index(),
        y="file",
        hue="phase",
        hue_order=reversed(["gcc (preprocessing 4)", 'cudafe++', 'gcc (compiling)',
                   "gcc (preprocessing 1)", 'cicc', 'ptxas',
                   'fatbinary',
                  ]),
        palette=palette,
        weights="fraction",
        multiple="stack",
        kind="hist",
        height=15,
    )
    plt.xlabel("fraction");
    plt.savefig(rel_out_path)
    print(f"-- Wrote relative compile time plot to {rel_out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        printf("""NVCC log analyzer

        Analyzes nvcc logs and outputs a figure with highest ranking translation
        units.

        Usage:
        python analyze_nvcc_log.py <nvcc_log_file.csv>
        cpp/scripts/analyze_nvcc_log.py <nvcc_log_file.csv>

        Generate the nvcc log file by adding:

        list(APPEND RAFT_CUDA_FLAGS "--time=CMakeFiles/nvcc_compile_log.csv")

        to cpp/cmake/modules/ConfigureCUDA.cmake.
        """)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Path {input_path} does not exist.")
    else:
        main(input_path)
