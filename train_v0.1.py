#!/usr/bin/env python3
"""
GeneMamba V0.1 training entry.
Usage:
  python train_v0.1.py                        # default: ../data
  python train_v0.1.py --data_dir Schmidt   # use Schmidt dataset
"""
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, help="Data directory (default: ../data)")
args, _ = parser.parse_known_args()

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

if args.data_dir:
    os.environ["GENE_DATA_DIR"] = args.data_dir
    print(f"[Data directory: {args.data_dir}]")
else:
    os.environ["GENE_DATA_DIR"] = "../data"

os.chdir(root_dir + "/train")
exec(open("run_training_v0.1.py").read())
