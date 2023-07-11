import argparse
import os
import subprocess
from urllib.request import urlretrieve

def get_dataset_path(name):
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", "%s.hdf5" % name)


def download_dataset(url, path):
    if not os.path.exists(path):
        # TODO: should be atomic
        print("downloading %s -> %s..." % (url, path))
        urlretrieve(url, path)


def convert_hdf5_to_fbin(path, normalize):
    if normalize and "angular" in path:
        p = subprocess.Popen(["python", "scripts/hdf5_to_fbin.py", "-n", "%s" % path])
    else:
        p = subprocess.Popen(["python", "scripts/hdf5_to_fbin.py", "%s" % path])
    p.wait()

def move(name, path):
    if "angular" in name:
        new_name = name.replace("angular", "inner")
    else:
        new_name = name
    new_path = os.path.join("data", new_name)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for bin_name in ["base.fbin", "query.fbin", "groundtruth.neighbors.ibin", "groundtruth.distances.fbin"]:
        os.rename("data/%s.%s" % (name, bin_name), "%s/%s" % (new_path, bin_name))


def download(name, normalize):
    path = get_dataset_path(name)
    try:
        url = "http://ann-benchmarks.com/%s.hdf5" % name
        download_dataset(url, path)

        convert_hdf5_to_fbin(path, normalize)

        move(name, path)
    except:
        print("Cannot download %s" % url)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", help="dataset to download", default="glove-100-angular")
    parser.add_argument("--normalize", help="normalize cosine distance to inner product", action="store_true")

    args = parser.parse_args()

    download(args.name, args.normalize)

if __name__ == "__main__":
    main()