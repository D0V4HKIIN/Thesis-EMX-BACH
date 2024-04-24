import color_print
import pathlib
import sys
import subprocess
import re
import math
import statistics
import numpy as np
from scipy import stats
import datetime
import os
import sys
import time

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
BIN_PATH = ROOT_PATH / "bin"
RES_PATH = ROOT_PATH / "res"
TEST_PATH = ROOT_PATH / "tests"
OUTPUT_PATH = TEST_PATH / "out" / "measurements"
CONFIG_PATH = ROOT_PATH / "tools" / "test_config.txt"

TEST_CASES = [
    ("test0", "test1"),
    ("testScience", "testTemplate"),
    ("ptf_m82_s_2k", "ptf_m82_t_2k"),
    ("sparse0", "sparse1"),
    ("ztf_m1_s_3k",  "ztf_m1_t_3k"),
    ("skyM-T-4k", "skyM-S-4k"),
    ("skyM-T-5k", "skyM-S-5k"),
    ("skyM-T-6k", "skyM-S-6k"),
    ("skyM-T-7k", "skyM-S-7k"),
    ("skyM-T-8k", "skyM-S-8k"),
    ("skyM-T-9k", "skyM-S-9k"),
    ("skyM-T-10k", "skyM-S-10k")
]

def run(binary, template_name, science_name, id, in_path, out_path):
    
    exe_path = BIN_PATH / binary / f"{binary}.exe"
    exe_args = [str(exe_path)]
    
    match binary:
        case "bach" | "xbach":
            exe_args.extend([
                "-ip", str(in_path),
                "-t", f"{template_name}.fits",
                "-s", f"{science_name}.fits",
                "-op", str(out_path / f"{binary}-{id}_"),
                "-v",
                "-vt"
            ])
        case "hotpants":
            exe_args.extend([
                "-inim", f"{str(in_path / science_name)}.fits",
                "-tmplim", f"{str(in_path / template_name)}.fits",
                "-outim", str(out_path / f"{binary}-{id}_out.fits"),
                "-oci", str(out_path / f"{binary}-{id}_conv.fits")
            ])
    
    with open(out_path / f"{binary}-{id}_out.txt", "w") as out_stream:
        if not subprocess.run(args=exe_args, stdout=out_stream, stderr=out_stream):
            print(f"{color_print.RED}Process exited with error status for binary {binary}.")

    time.sleep(500 / 1000)

time_matcher = re.compile(r".*took (?:(\d+) ?s )?(\d+) ?ms")
def measure_execution_time(binary, out_path, external_path):
    for i, (template_name, science_name) in enumerate(TEST_CASES):
        in_path = RES_PATH
        if not ((in_path / f"{template_name}.fits").exists() or\
                (in_path / f"{ science_name}.fits").exists()):
            if external_path == None: break
            elif (external_path / f"{template_name}.fits").exists() and\
                 (external_path / f"{science_name}.fits").exists():
                in_path = external_path
            else:
                print("Bad external path", file=sys.stderr)
                exit(1)

        runs = []
        filename=f"{binary}-t{i+1}.txt"
        print(filename)
        run(binary, template_name, science_name, i+1, in_path, out_path)
        for _ in range(10): 
            run(binary, template_name, science_name, i+1, in_path, out_path)
            
            times=[]
            with open(out_path / f"{binary}-{i+1}_out.txt", "r") as run_log:
                for line in run_log.readlines():
                    matches = time_matcher.findall(line)
                    if len(matches) == 0: continue

                    match matches[0]:
                        case (s_str, ms_str) if s_str != '':
                            times.append(int(s_str)*1000 + int(ms_str))
                        case (s_str, ms_str) if s_str == '':
                            times.append(int(ms_str))

            if binary in { "xbach", "bach" } and len(times) != 9 or binary == "hotpants" and len(times) != 1:
                print(f"{color_print.YELLOW}Ignoring failed run...")
                continue

            runs.append(times)
            print(times)
        
        if len(runs[0]) > 1:
            step_times={"Ini":[], "SSS":[], "CMV":[], "CD":[], "KSC":[],
                    "Conv":[], "Sub":[], "Fin":[], "Total":[]}
            steps = ["Ini", "SSS", "CMV", "CD", "KSC", "Conv", "Sub", "Fin", "Total"]
            for run_ in runs:
                for j, t in enumerate(run_):
                    step_times[steps[j]].append(t)
                    
            with open(out_path / f"{binary}-t{i+1}.txt", "w") as time_log:
                for step in steps:
                    time_log.write(f"{step}:")
                    for time in step_times[step]:
                        time_log.write(f" {time}")
                    time_log.write("\n")
        else:
            with open(out_path / f"{binary}-t{i+1}.txt", "w") as time_log:
                time_log.write("Total:")
                for run_ in runs:    
                    for time in run_:
                        time_log.write(f" {time}")
                time_log.write("\n")

        
        
def main(args):
    color_print.init()
    date = str(datetime.datetime.now().date())

    external_path = None

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as input:
            path_str = input.readline().strip()
            external_path = pathlib.Path(path_str)

        print(f"{color_print.CYAN}Using external path: {external_path}")
    else:
        print(f"{color_print.YELLOW}No test config file found. Some tests will be skipped.")
        print(f"{color_print.YELLOW}Please create {CONFIG_PATH} and put the path to the externel test files.")

    n = 0
    if (OUTPUT_PATH / date).exists():
        n = len(os.listdir((OUTPUT_PATH / date)))
    
    (OUTPUT_PATH / date / str(n)).mkdir(parents=True, exist_ok=True)
    path = (OUTPUT_PATH / date) / str(n)
    measure_execution_time("bach", path, external_path)
    measure_execution_time("xbach", path, external_path)
    measure_execution_time("hotpants", path, external_path)
    color_print.destroy()

if __name__ == "__main__": main(sys.argv)
