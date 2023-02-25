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
use File::Find;
use Getopt::Std;

my $QPS = 'QPS';
my $AVG_LATENCY = 'avg_latency(ms)';
my $P99_LATENCY = 'p99_latency(ms)';
my $P999_LATENCY = 'p999_latency(ms)';
my @CONDITIONS = ([$QPS, 2000], ['recall', 0.9], ['recall', 0.95]);


my $USAGE = << 'END';
usage: [-f] [-l avg|p99|p999] [-o output.csv] groundtruth.neighbors.ibin result_paths...
  result_paths... are paths to the search result files.
    Can specify multiple paths.
    For each of them, if it's a directory, all the .txt files found under
    it recursively will be regarded as inputs.

  -f: force to recompute recall and update it in result file if needed
  -l: output search latency rather than QPS. Available options:
        "avg" for average latency;
        "p99" for 99th percentile latency;
        "p999" for 99.9th percentile latency.
  -o: also write result to a csv file
END


my %opt;
getopts('fl:o:', \%opt)
  or die $USAGE;
my $force_calc_recall = exists $opt{f} ? 1 : 0;
my $csv_file;
$csv_file = $opt{o} if exists $opt{o};
my $metric = $QPS;
if (exists $opt{l}) {
    my $option = $opt{l};
    if ($option eq 'avg') {
        $metric = $AVG_LATENCY;
    }
    elsif ($option eq 'p99') {
        $metric = $P99_LATENCY;
    }
    elsif ($option eq 'p999') {
        $metric = $P999_LATENCY;
    }
    else {
        die
          "[error] illegal value for '-l': '$option'. Must be 'avg', 'p99' or 'p999'\n";
    }
}

@ARGV >= 2
  or die $USAGE;


my $truth_file = shift @ARGV;
my ($k, $dataset, $distance, $results) = get_all_results($metric, @ARGV);
if (!defined $k) {
    print STDERR "no result file found\n";
    exit -1;
}
print STDERR "dataset = $dataset, distance = $distance, k = $k\n\n";
calc_missing_recall($results, $truth_file, $force_calc_recall);

my @results = sort {
         $a->{name} cmp $b->{name}
      or $a->{recall} <=> $b->{recall}
      or $b->{qps} <=> $a->{qps}
} @$results;
printf("%-60s  %6s %16s  %s\n", '', 'Recall', $metric, 'search_param');
for my $result (@results) {
    my $fmt = ($metric eq $QPS) ? '%16.1f' : '%16.3f';
    my $qps = $result->{qps};
    $qps *= 1000 if $metric ne $QPS;    # the unit of latency is ms
    printf("%-60s  %6.4f ${fmt}  %s\n",
        $result->{name}, $result->{recall}, $qps, $result->{search_param});
}
if (defined $csv_file) {
    open my $fh, '>', $csv_file;
    print {$fh} ",Recall,${metric},search_param\n";
    for my $result (@results) {
        my $qps = $result->{qps};
        $qps *= 1000 if $metric ne $QPS;
        printf {$fh} (
            "%s,%.4f,%.3f,%s\n", $result->{name}, $result->{recall},
            $qps, $result->{search_param}
        );
    }
}
print "\n";
calc_and_print_estimation($results, $metric, \@CONDITIONS);




sub read_result {
    my ($fname) = @_;
    open my $fh, '<', $fname;
    my %attr;
    while (<$fh>) {
        chomp;
        next if /^\s*$/;
        my $pos = index($_, ':');
        $pos != -1
          or die "[error] no ':' is found: '$_'\n";
        my $key = substr($_, 0, $pos);
        my $val = substr($_, $pos + 1);
        $key =~ s/^\s+|\s+$//g;
        $val =~ s/^\s+|\s+$//g;

        # old version benchmark compatible
        if ($key eq 'search_time') {
            $key = 'average_search_time';
            $val *= $attr{batch_size};
        }
        $attr{$key} = $val;
    }
    return \%attr;
}

sub overwrite_recall_to_result {
    my ($fname, $recall) = @_;
    open my $fh_in, '<', $fname;
    $recall = sprintf("%f", $recall);
    my $out;
    while (<$fh_in>) {
        s/^recall: .*/recall: $recall/;
        $out .= $_;
    }
    close $fh_in;

    open my $fh_out, '>', $fname;
    print {$fh_out} $out;
}

sub append_recall_to_result {
    my ($fname, $recall) = @_;
    open my $fh, '>>', $fname;
    printf {$fh} ("recall: %f\n", $recall);
}

sub get_all_results {
    my ($metric) = shift @_;

    my %fname;
    my $wanted = sub {
        if (-f && /\.txt$/) {
            $fname{$File::Find::name} = 1;
        }
    };
    find($wanted, @_);

    my $k;
    my $dataset;
    my $distance;
    my @results;
    for my $f (sort keys %fname) {
        print STDERR "reading $f ...\n";
        my $attr = read_result($f);
        if (!defined $k) {
            $k = $attr->{k};
            $dataset = $attr->{dataset};
            $distance = $attr->{distance};
        }
        else {
            $attr->{k} eq $k
              or die "[error] k should be $k, but is $attr->{k} in $f\n";
            $attr->{dataset} eq $dataset
              or die
              "[error] dataset should be $dataset, but is $attr->{dataset} in $f\n";
            $attr->{distance} eq $distance
              or die
              "[error] distance should be $distance, but is $attr->{distance} in $f\n";
        }

        my $batch_size = $attr->{batch_size};
        $batch_size =~ s/000000$/M/;
        $batch_size =~ s/000$/K/;
        my $search_param = $attr->{search_param};
        $search_param =~ s/^{//;
        $search_param =~ s/}$//;
        $search_param =~ s/,/ /g;
        $search_param =~ s/"//g;

        my $qps;
        if ($metric eq $QPS) {
            $qps = $attr->{batch_size} / $attr->{average_search_time};
        }
        elsif ($metric eq $AVG_LATENCY) {
            $qps = $attr->{average_search_time};
        }
        elsif ($metric eq $P99_LATENCY) {
            exists $attr->{p99_search_time}
              or die "[error] p99_search_time is not found\n";
            $qps = $attr->{p99_search_time};
        }
        elsif ($metric eq $P999_LATENCY) {
            exists $attr->{p999_search_time}
              or die "[error] p999_search_time is not found\n";
            $qps = $attr->{p999_search_time};
        }
        else {
            die "[error] unknown latency type: '$metric'\n";
        }
        my $result = {
            file => $f,
            name => "$attr->{name}-batch${batch_size}",
            search_param => $search_param,
            qps => $qps,
        };

        if (exists $attr->{recall}) {
            $result->{recall} = $attr->{recall};
        }
        push @results, $result;
    }
    return $k, $dataset, $distance, \@results;
}

sub read_ibin {
    my ($fname) = @_;

    open my $fh, '<:raw', $fname;
    my $raw;

    read($fh, $raw, 8);
    my ($nrows, $dim) = unpack('LL', $raw);

    my $expected_size = 8 + $nrows * $dim * 4;
    my $size = (stat($fh))[7];
    $size == $expected_size
      or die(
        "[error] expected size is $expected_size, but actual size is $size\n");

    read($fh, $raw, $nrows * $dim * 4) == $nrows * $dim * 4
      or die "[error] read $fname failed\n";
    my @data = unpack('l' x ($nrows * $dim), $raw);
    return \@data, $nrows, $dim;
}

sub pick_k_neighbors {
    my ($neighbors, $nrows, $ncols, $k) = @_;

    my @res;
    for my $i (0 .. $nrows - 1) {
        my %neighbor_set;
        for my $j (0 .. $k - 1) {
            $neighbor_set{$neighbors->[$i * $ncols + $j]} = 1;
        }
        push @res, \%neighbor_set;
    }
    return \@res;
}


sub calc_recall {
    my ($truth_k_neighbors, $result_neighbors, $nrows, $k) = @_;

    my $recall = 0;
    for my $i (0 .. $nrows - 1) {
        my $tp = 0;
        for my $j (0 .. $k - 1) {
            my $neighbor = $result_neighbors->[$i * $k + $j];
            ++$tp if exists $truth_k_neighbors->[$i]{$neighbor};
        }
        $recall += $tp;
    }
    return $recall / $k / $nrows;
}

sub calc_missing_recall {
    my ($results, $truth_file, $force_calc_recall) = @_;

    my $need_calc_recall = grep { !exists $_->{recall} } @$results;
    return unless $need_calc_recall || $force_calc_recall;

    my ($truth_neighbors, $nrows, $truth_k) = read_ibin($truth_file);
    $truth_k >= $k
      or die "[error] ground truth k ($truth_k) < k($k)\n";
    my $truth_k_neighbors =
      pick_k_neighbors($truth_neighbors, $nrows, $truth_k, $k);

    for my $result (@$results) {
        next if exists $result->{recall} && !$force_calc_recall;

        my $result_bin_file = $result->{file};
        $result_bin_file =~ s/txt$/ibin/;
        print STDERR "calculating recall for $result_bin_file ...\n";
        my ($result_neighbors, $result_nrows, $result_k) =
          read_ibin($result_bin_file);
        $result_k == $k
          or die
          "[error] k should be $k, but is $result_k in $result_bin_file\n";
        $result_nrows == $nrows
          or die
          "[error] #row should be $nrows, but is $result_nrows in $result_bin_file\n";

        my $recall =
          calc_recall($truth_k_neighbors, $result_neighbors, $nrows, $k);
        if (exists $result->{recall}) {
            my $new_value = sprintf("%f", $recall);
            if ($result->{recall} ne $new_value) {
                print "update recall: $result->{recall} -> $new_value\n";
                overwrite_recall_to_result($result->{file}, $recall);
            }
        }
        else {
            append_recall_to_result($result->{file}, $recall);
        }
        $result->{recall} = $recall;
    }
}


sub estimate {
    my ($results, $condition, $value) = @_;
    my %point_of;
    for my $result (@$results) {
        my $point;
        if ($condition eq 'recall') {
            $point = [$result->{recall}, $result->{qps}];
        }
        else {
            $point = [$result->{qps}, $result->{recall}];
        }
        push @{$point_of{$result->{name}}}, $point;
    }

    my @names = sort keys %point_of;
    my @result;
    for my $name (@names) {
        my @points = sort { $a->[0] <=> $b->[0] } @{$point_of{$name}};
        if ($value < $points[0][0] || $value > $points[$#points][0]) {
            push @result, -1;
            next;
        }
        elsif ($value == $points[0][0]) {
            push @result, $points[0][1];
            next;
        }

        for my $i (1 .. $#points) {
            if ($points[$i][0] >= $value) {
                push @result,
                  linear_interpolation($value, @{$points[$i - 1]},
                    @{$points[$i]});
                last;
            }
        }
    }
    return \@names, \@result;
}

sub linear_interpolation {
    my ($x, $x1, $y1, $x2, $y2) = @_;
    return $y1 + ($x - $x1) * ($y2 - $y1) / ($x2 - $x1);
}

sub merge {
    my ($all, $new, $scale) = @_;
    @$all == @$new
      or die "[error] length is not equal\n";
    for my $i (0 .. @$all - 1) {
        push @{$all->[$i]}, $new->[$i] * $scale;
    }
}

sub calc_and_print_estimation {
    my ($results, $metric, $conditions) = @_;

    my @conditions = grep {
        my $target = $_->[0];
        if ($target eq 'recall' || $target eq $metric) {
            1;
        }
        else {
                 $target eq $QPS
              || $target eq $AVG_LATENCY
              || $target eq $P99_LATENCY
              || $target eq $P999_LATENCY
              or die "[error] unknown condition: '$target'\n";
            0;
        }
    } @$conditions;

    my @headers = map {
        my $header;
        if ($_->[0] eq 'recall') {
            $header = $metric . '@recall' . $_->[1];
        }
        elsif ($_->[0] eq $metric) {
            $header = 'recall@' . $metric . $_->[1];
        }
        $header;
    } @conditions;

    my $scale = ($metric eq $QPS) ? 1 : 1000;
    my $estimations;
    for my $condition (@conditions) {
        my ($names, $estimate) = estimate($results, @$condition);
        if (!defined $estimations) {
            @$estimations = map { [$_] } @$names;
        }
        merge($estimations, $estimate, $scale);
    }

    my $fmt = "%-60s" . ("  %16s" x @headers) . "\n";
    printf($fmt, '', @headers);
    $fmt =~ s/16s/16.4f/g;
    for (@$estimations) {
        printf($fmt, @$_);
    }
}
