import color_print
import datetime
import pathlib
import sys
import subprocess
import re
import numpy as np
import datetime
import os
import sys

from test_generator import *


def run(
    test_string,
    binary,
    template_name,
    science_name,
    in_path,
    out_path,
    num_cores,
    cpu_part,
    accelerator_part,
):

    exe_path = BIN_PATH / binary / binary
    exe_args = [str(exe_path)]

    match binary:
        case "bach" | "xbach" | "emxbach":
            exe_args.extend(
                [
                    "-ip",
                    str(in_path),
                    "-t",
                    f"{template_name}.fits",
                    "-s",
                    f"{science_name}.fits",
                    "-op",
                    f"{out_path / test_string}_",
                    "-v",
                    "-vt",
                    "--cpuPart",
                    str(cpu_part),
                    "--accelerators",
                    f"{ACCELERATOR_PLATFORM}:{ACCELERATOR_DEVICE}:{accelerator_part}",
                ]
            )
        case "hotpants":
            exe_args.extend(
                [
                    "-inim",
                    f"{str(in_path / science_name)}.fits",
                    "-tmplim",
                    f"{str(in_path / template_name)}.fits",
                    "-outim",
                    f"{str(out_path / test_string)}_out.fits",
                    "-oci",
                    f"{str(out_path / test_string)}_conv.fits",
                ]
            )

    env = os.environ.copy()
    if NUM_CORES:
        env["OMP_NUM_THREADS"] = str(num_cores)

    with open(out_path / f"{test_string}_out.txt", "w") as out_stream:
        if not subprocess.run(
            args=exe_args, stdout=out_stream, stderr=out_stream, env=env
        ):
            print(
                f"{color_print.RED}Process exited with error status for binary {binary}."
            )


# makes 10 measurements
def measure_n(
    filename,
    binary,
    template_name,
    science_name,
    in_path,
    out_path,
    num_cores,
    cpu_part,
    accelerator_part,
):
    runs = []
    for _ in range(10):
        run(
            filename,
            binary,
            template_name,
            science_name,
            in_path,
            out_path,
            num_cores,
            cpu_part,
            accelerator_part,
        )

        times = []
        with open(out_path / f"{filename}_out.txt", "r") as run_log:
            for line in run_log.readlines():
                matches = time_matcher.findall(line)
                if len(matches) == 0:
                    continue

                match matches[0]:
                    case (s_str, ms_str) if s_str != "":
                        times.append(int(s_str) * 1000 + int(ms_str))
                    case (s_str, ms_str) if s_str == "":
                        times.append(int(ms_str))
            # print log if run failed
            if binary in {"emxbach", "xbach", "bach"} and len(times) != 11:
                run_log.seek(0)
                print(run_log.read())

        if (
            binary in {"emxbach", "xbach", "bach"}
            and len(times) != 11
            or binary == "hotpants"
            and len(times) != 1
        ):
            print(len(times))
            print(f"{color_print.YELLOW}Ignoring failed run...")
            continue

        runs.append(times)
        print(times)
    return runs


time_matcher = re.compile(r".*took (?:(\d+) ?s )?(\d+) ?ms")


def measure_execution_time(tests, out_path, external_path):
    start = datetime.datetime.now()
    for i, test in enumerate(tests):
        binary = test["software"]
        science_name = TEST_CASES[test["test"] - 1][0]
        template_name = TEST_CASES[test["test"] - 1][1]
        cpu_part = test["cpu"]
        accelerator_part = test["accelerator"]

        in_path = RES_PATH
        if not (
            (in_path / f"{template_name}.fits").exists()
            or (in_path / f"{ science_name}.fits").exists()
        ):
            if external_path == None:
                break
            elif (external_path / f"{template_name}.fits").exists() and (
                external_path / f"{science_name}.fits"
            ).exists():
                in_path = external_path
            else:
                print("Bad external path", file=sys.stderr)
                exit(1)

        # run to compile shaders and cache files
        run(
            "",
            binary,
            template_name,
            science_name,
            in_path,
            out_path,
            NUM_CORES,
            cpu_part,
            accelerator_part,
        )

        cores = test["core"]

        filename = test_filename(test)
        now = datetime.datetime.now()
        tests_remaining = len(tests) - i
        print(
            filename,
            ".txt tests remaining: ",
            tests_remaining,
            " ETA: ",
            (now - start) / i * tests_remaining if i != 0 else "-",
            sep="",
        )

        runs = measure_n(
            filename,
            binary,
            template_name,
            science_name,
            in_path,
            out_path,
            cores,
            cpu_part,
            accelerator_part,
        )

        if len(runs[0]) > 1:
            step_times = {
                "Ini": [],
                "SSS": [],
                "CMV": [],
                "CD": [],
                "KSC": [],
                "MakeKernels": [],
                "Convolution": [],
                "Conv": [],
                "Sub": [],
                "Fin": [],
                "Total": [],
            }
            steps = [
                "Ini",
                "SSS",
                "CMV",
                "CD",
                "KSC",
                "MakeKernels",
                "Convolution",
                "Conv",
                "Sub",
                "Fin",
                "Total",
            ]
            for run_ in runs:
                for j, t in enumerate(run_):
                    step_times[steps[j]].append(t)

            with open(out_path / f"{filename}.txt", "w") as time_log:
                for step in steps:
                    time_log.write(f"{step}:")
                    for time in step_times[step]:
                        time_log.write(f" {time}")
                    time_log.write("\n")
        else:
            with open(out_path / f"{filename}.txt", "w") as time_log:
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
        print(
            f"{color_print.YELLOW}No test config file found. Some tests will be skipped."
        )
        print(
            f"{color_print.YELLOW}Please create {CONFIG_PATH} and put the path to the externel test files."
        )

    n = 0
    if (OUTPUT_PATH / date).exists():
        n = len(os.listdir((OUTPUT_PATH / date)))

    (OUTPUT_PATH / date / str(n)).mkdir(parents=True, exist_ok=True)
    out_path = (OUTPUT_PATH / date) / str(n)

    # tests = generate_tests()
    # tests = generate_arg_tests()
    tests = get_tests_from_args(args)

    print("measuring", len(tests), "data points")

    measure_execution_time(tests, out_path, external_path)

    print("output is in", out_path)
    color_print.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
