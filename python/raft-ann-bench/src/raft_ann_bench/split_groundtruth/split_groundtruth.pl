#!/usr/bin/perl

# =============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

use warnings;
use strict;
use autodie qw(open close);


@ARGV == 2
  or die "usage: $0 input output_prefix\n";

open my $fh, '<:raw', $ARGV[0];

my $raw;
read($fh, $raw, 8);
my ($nrows, $dim) = unpack('LL', $raw);

my $expected_size = 8 + $nrows * $dim * (4 + 4);
my $size = (stat($fh))[7];
$size == $expected_size
  or die("error: expected size is $expected_size, but actual size is $size\n");


open my $fh_out1, '>:raw', "$ARGV[1].neighbors.ibin";
open my $fh_out2, '>:raw', "$ARGV[1].distances.fbin";

print {$fh_out1} $raw;
print {$fh_out2} $raw;

read($fh, $raw, $nrows * $dim * 4);
print {$fh_out1} $raw;
read($fh, $raw, $nrows * $dim * 4);
print {$fh_out2} $raw;
