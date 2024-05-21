import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import glob
import argparse
import os

def run(yaml_path):
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

def run_analysis_cfg(yaml_path):
    """Function to execute a script using subprocess."""
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', 'run_analysis_cfg.py', yaml_path], check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")

def run_analysis_cfg2(yaml_path):
    """Function to execute a script using subprocess."""
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', 'run_analysis_cfg2.py', yaml_path], check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")

def run_analysis_regress(yaml_path):
    """Function to execute a script using subprocess."""
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', 'run_analysis_regress.py', yaml_path], check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")

def calc_mse_pix(yaml_path):
    """Function to execute a script using subprocess."""
    try:
        # Ensure your script has the appropriate executable permissions
        result = subprocess.run(['python3', 'calc_mse_pix.py', yaml_path], check=True, text=True, capture_output=True)
        print(f"{yaml_path} ran successfully:")
    except subprocess.CalledProcessError as e:
        error_message=e.stderr.decode()
        print(f"Error running {yaml_path}: {error_message}:")

def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple scripts in parallel")
    parser.add_argument("folfile", type=str, help="Folder/File containing the yamls")
    parser.add_argument("--n-par", type=int, default=0, help="Number of parallel processes")
    parser.add_argument("--exname", type=str,default="run", help="exname.py will be ran, default is \"run\" ")
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
        if args.exname=="run":
            results = executor.map(run, yaml_paths)
        elif args.exname=="run_analysis":
            results = executor.map(run_analysis, yaml_paths)
        elif args.exname=="run_analysis_cfg":
            results = executor.map(run_analysis_cfg, yaml_paths)
        elif args.exname=="run_analysis_cfg2":
            results = executor.map(run_analysis_cfg2, yaml_paths)
        elif args.exname=="run_analysis_regress":
            results = executor.map(run_analysis_regress, yaml_paths)
        elif args.exname=="calc_mse_pix":
            results = executor.map(calc_mse_pix, yaml_paths)
        else:
            raise ValueError("exname not recognized")
    print("Done")