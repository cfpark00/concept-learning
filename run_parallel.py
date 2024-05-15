import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import glob
import argparse
import os

def run_yaml(yaml_path):
    """Function to execute a script using subprocess."""
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', 'run.py', yaml_path], check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")

def run_analysis(yaml_path):
    """Function to execute a script using subprocess."""
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', 'run_analysis.py', yaml_path], check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple scripts in parallel")
    parser.add_argument("folfile", type=str, help="Folder/File containing the yamls")
    parser.add_argument("--n-par", type=int, default=0, help="Number of parallel processes")
    parser.add_argument("--analysis", action="store_true", help="Run analysis")
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments
    args=parse_args()
    #check if folfile is a file
    if os.path.isfile(args.folfile):
        with open(args.folfile,"r") as f:
            yaml_paths = f.readlines()
        yaml_paths = [x.strip() for x in yaml_paths]
    else:
        fol=args.folfile
        yaml_paths = glob.glob(f"{fol}/*.yaml")
    if args.n_par>0:
        n_par=args.n_par
    else:
        n_par=len(yaml_paths)
    print("Running:",yaml_paths)

    # Use ThreadPoolExecutor to run scripts in parallel
    with ThreadPoolExecutor(max_workers=n_par) as executor:
        # Map each script to the executor
        if args.analysis:
            results = executor.map(run_analysis, yaml_paths)
        else:
            results = executor.map(run_yaml, yaml_paths)
    print("Done")