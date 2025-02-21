import color_print
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
from astropy.io import fits

TEST_TABLE = [
    # ID | Fast? | External? | Science       | Template      | HOTPANTS conv    | HOTPANTS sub    | Max abs error S,T | Max rel error S,T
    ( 1,   True,   False,     "test0",        "test1",        "test01_conv",     "test01_sub",     (2e-4, 5e-4),       (5e-3, 4e-3)),
    ( 2,   True,   False,     "testScience",  "testTemplate", "testST_conv",     "testST_sub",     (8e-3, 2e-3),       (5e-6, 9e-1)),
    ( 3,   True,   False,     "ptf_m82_s_2k", "ptf_m82_t_2k", "ptf_m82_2k_conv", "ptf_m82_2k_sub", (2e-1, 3e0),        (1e-4, 4e-1)),
    ( 4,   False,  False,     "sparse0",      "sparse1",      "sparse01_conv",   "sparse01_sub",   (2e1,  5e0),        (3e-4, 5e-4)),
    ( 5,   True,   False,     "ztf_m1_s_3k",  "ztf_m1_t_3k",  "ztf_m1_3k_conv",  "ztf_m1_3k_sub",  (1e-3, 1e-2),       (1e-6, 1e0)),
    ( 6,   True,   True,      "skyM-S-4k",    "skyM-T-4k",    "skyM-4k_conv",    "skyM-4k_sub",    (1e-2, 1e-2),       (1e-1, 1e-1)),
    ( 7,   False,  True,      "skyM-S-5k",    "skyM-T-5k",    "skyM-5k_conv",    "skyM-5k_sub",    (1e-2, 1e-2),       (1e-6, 3e0)),
    ( 8,   False,  True,      "skyM-S-6k",    "skyM-T-6k",    "skyM-6k_conv",    "skyM-6k_sub",    (1e-2, 1e-2),       (1e-6, 3e0)),
    ( 9,   False,  True,      "skyM-S-7k",    "skyM-T-7k",    "skyM-7k_conv",    "skyM-7k_sub",    (1e-2, 1e-2),       (1e-6, 3e0)),
    (10,   False,  True,      "skyM-S-8k",    "skyM-T-8k",    "skyM-8k_conv",    "skyM-8k_sub",    (1e-2, 1e-2),       (1e-6, 3e0)),
    (11,   False,  True,      "skyM-S-9k",    "skyM-T-9k",    "skyM-9k_conv",    "skyM-9k_sub",    (1e-2, 1e-2),       (1e-6, 3e0)),
    (12,   False,  True,      "skyM-S-10k",   "skyM-T-10k",   "skyM-10k_conv",   "skyM-10k_sub",   (1e-2, 1e-2),       (1e-6, 3e0))
]

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
BIN_PATH = ROOT_PATH / "bin"
RES_PATH = ROOT_PATH / "res"
TEST_PATH = ROOT_PATH / "tests"
OUTPUT_PATH = TEST_PATH / "out"
CONFIG_PATH = ROOT_PATH / "tools" / "test_config.txt"

def diff_fits(h_path, b_path):
    print("diffing", h_path, "and", b_path)
    h_file = fits.open(h_path)
    b_file = fits.open(b_path)
    
    assert(len(h_file) == 1)
    assert(len(b_file) == 1)

    h_data = h_file[0].data
    b_data = b_file[0].data
    
    assert(h_data.ndim == 2)
    assert(b_data.ndim == 2)

    max_error_abs = -10000000
    max_error_rel = -10000000
    abs_coords = (-1, -1)
    mean_error_abs = 0
    mean_error_rel = 0
    wrong_nans = 0
    count = 0
    
    for x in range(len(h_data)):
        for y in range(len(h_data[x])):
            h = h_data[x, y]
            b = b_data[x, y]

            if math.isnan(h) or math.isnan(b):
                if math.isnan(h) != math.isnan(b):
                    wrong_nans += 1
                
                continue
            
            # Absolute
            error_abs = abs(h - b)

            mean_error_abs += error_abs

            if error_abs > max_error_abs:
                max_error_abs = error_abs
                abs_coords = (x, y)

            # Relative
            if h > 0:
                error_rel = error_abs / h
                
                mean_error_rel += error_rel

                if error_rel > max_error_rel:
                    max_error_rel = error_rel

            count += 1

    mean_error_abs /= count
    mean_error_rel /= count

    h_file.close()
    b_file.close()

    return max_error_abs, mean_error_abs, max_error_rel, mean_error_rel, wrong_nans, abs_coords

def run_test(test_index, verbose, build_config, external_path):
    (id, _, external, science_name, template_name, conv_name, sub_name, max_abs_error, max_rel_error) = TEST_TABLE[test_index]
    res_path = external_path if external else RES_PATH

    print(f"{color_print.CYAN}Running test {id}...")

    exe_path = BIN_PATH / "xbach"

    exe_args = [str(exe_path)]
    exe_args += ["-ip", str(res_path)]
    exe_args += ["-s", f"{science_name}.fits"]
    exe_args += ["-t", f"{template_name}.fits"]
    exe_args += ["-op", str(OUTPUT_PATH / f"test{id}_")]
    exe_args += ["-vt"]

    start_time = time.time()

    with open(OUTPUT_PATH / f"test{id}_out.txt", "w") as out_stream:
        if not subprocess.run(args=exe_args, stdout=out_stream, stderr=out_stream):
            print(f"{color_print.RED}X-BACH exited with an error code!")
            return False

    end_time = time.time()
    test_time = end_time - start_time

    print(f"Test took {test_time:.2f} seconds")

    conv_out_path = OUTPUT_PATH / f"test{id}_diff.fits"
    sub_out_path = OUTPUT_PATH / f"test{id}_sub.fits"

    if not conv_out_path.exists() or not sub_out_path.exists():
        print(f"{color_print.RED}At least one X-BACH output is missing. The program did not run correctly.")
        return False

    base_test_path = external_path if external else TEST_PATH

    conv_max_abs_err, conv_mean_abs_err, conv_max_rel_err, conv_mean_rel_err, conv_wrong_nans, conv_max_coords = diff_fits(base_test_path / f"{conv_name}.fits", conv_out_path)
    sub_max_abs_err, sub_mean_abs_err, sub_max_rel_err, sub_mean_rel_err, sub_wrong_nans, sub_max_coords = diff_fits(base_test_path / f"{sub_name}.fits", sub_out_path)
    print(f"Convolution errors: {conv_max_abs_err:.2e} (max abs)  {conv_max_rel_err:.2e} (max rel)")
    print(f"                    {conv_mean_abs_err:.2e} (mean abs) {conv_mean_rel_err:.2e} (mean rel)")
    print(f"                    {conv_wrong_nans} (NaN)")

    if verbose:
        print(f"                    Max abs error at ({conv_max_coords[0]}; {conv_max_coords[1]})")

    print(f"Subtraction errors: {sub_max_abs_err:.2e} (max abs)  {sub_max_rel_err:.2e} (max rel)")
    print(f"                    {sub_mean_abs_err:.2e} (mean abs) {sub_mean_rel_err:.2e} (mean rel)")
    print(f"                    {sub_wrong_nans} (NaN)")

    if verbose:
        print(f"                    Max abs error at ({sub_max_coords[0]}; {sub_max_coords[1]})")

    print(conv_max_abs_err, max_abs_error[0], conv_max_rel_err, max_rel_error[0], sub_max_abs_err, max_abs_error[1], sub_max_rel_err, max_rel_error[1])

    return conv_max_abs_err < max_abs_error[0] and conv_max_rel_err < max_rel_error[0] and\
        sub_max_abs_err < max_abs_error[1] and sub_max_rel_err < max_rel_error[1] and\
        conv_wrong_nans == 0 and sub_wrong_nans == 0

def print_help():
    print(f"{color_print.YELLOW}Usage: {pathlib.Path(__file__).name} [<flags>...]")
    print(f"{color_print.YELLOW}Possible flags:")
    print(f"{color_print.YELLOW}-h: Print this help text and exits.")
    print(f"{color_print.YELLOW}-v: Verbose printing.")
    print(f"{color_print.YELLOW}--all: Runs all tests (default).")
    print(f"{color_print.YELLOW}--fast: Runs all tests which are slow.")
    print(f"{color_print.YELLOW}--slow: Runs all tests which are fast.")
    print(f"{color_print.YELLOW}--external: Runs all tests which are external.")
    print(f"{color_print.YELLOW}--release: Runs the program in release mode. If not specified, the debug build is used.")
    print(f"{color_print.YELLOW}--generate: Generates the conv and sub files in the test folder")

def run_tests(verbose, tests, build_config, external_path):
    print(f"{color_print.CYAN}Running X-BACH ({build_config}) from \"{BIN_PATH.resolve()}\"")

    print()
    print(f"There are a total of {len(tests)} tests to run:")

    for i in tests:
        test = TEST_TABLE[i]
        print(f" Test {test[0]}, Science: {test[3]} Template: {test[4]}")

    print()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Clear out the output directory before running any tests
    for root, dirs, files in os.walk(OUTPUT_PATH):
        for f in files:
            os.unlink(os.path.join(root, f))

        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    failed_tests = 0
    total_tests = 0

    start_time = time.time()

    for i in tests:
        test_id = TEST_TABLE[i][0]
        test_success = run_test(i, verbose, build_config, external_path)
        
        if test_success:
            print(f"{color_print.GREEN}Test {test_id} succeeded!")
        else:
            failed_tests += 1
            print(f"{color_print.RED}Test {test_id} failed!")

        total_tests += 1
        print()
    
    end_time = time.time()
    tess_time = end_time - start_time

    if failed_tests > 0:
        print(f"{color_print.RED}{failed_tests} / {total_tests} tests failed!")

        return False

    print(f"{color_print.GREEN}All tests were successful!")
    print(f"Tests took {tess_time:.2f} seconds")

def run(binary, template_name, science_name, conv_name, sub_name, in_path, out_path):
    
    exe_path = BIN_PATH / binary
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
                "-outim", str(out_path / f"{sub_name}.fits"),
                "-oci", str(out_path / f"{conv_name}.fits")
            ])
    
    with open(out_path / f"{science_name}.txt", "w") as out_stream:
        if not subprocess.run(args=exe_args, stdout=out_stream, stderr=out_stream):
            print(f"{color_print.RED}Process exited with error status for binary {binary}.")

def main(args):
    color_print.init()

    external_path = None

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as input:
            path_str = input.readline().strip()
            external_path = pathlib.Path(path_str)

        print(f"{color_print.CYAN}Using external path: {external_path}")
    else:
        print(f"{color_print.YELLOW}No test config file found. Some tests will be skipped.")
        print(f"{color_print.YELLOW}Please create {CONFIG_PATH} and put the path to the externel test files.")

    # Parse args
    verbose = False
    debug = True
    generate = False
    tests = []

    for arg in args:
        if arg == "-h":
            print_help()
            return True
        elif arg == "-v":
            verbose = True
        elif arg == "--all":
            tests += [i for i in range(len(TEST_TABLE))]
        elif arg == "--fast":
            tests += [i for i in range(len(TEST_TABLE)) if TEST_TABLE[i][1]]
        elif arg == "--slow":
            tests += [i for i in range(len(TEST_TABLE)) if not TEST_TABLE[i][1]]
        elif arg == "--external":
            tests += [i for i in range(len(TEST_TABLE)) if TEST_TABLE[i][2]]
        elif arg == "--release":
            debug = False
        elif arg == "--generate":
            generate = True
        else:
            print(f"{color_print.RED}Unrecognized flag: {arg}")
            print()
            print_help()
            return False

    if len(tests) == 0:
        tests = [i for i in range(len(TEST_TABLE))]

    if external_path is None:
        new_tests = []
        removed_tests = []
        
        for i in tests:
            if TEST_TABLE[i][2]:
                removed_tests.append(i)
            else:
                new_tests.append(i)

        tests = new_tests

        if len(removed_tests) > 0:
            print()
            print(f"{color_print.YELLOW}The following tests were skipped due to missing test config:")

            for i in removed_tests:
                print(f"{color_print.YELLOW}  Test {TEST_TABLE[i][0]}")

    build_config = "Debug" if debug else "Release"

    if generate:
        for i in tests:
            (id, _, external, science_name, template_name, conv_name, sub_name, max_abs_error, max_rel_error) = TEST_TABLE[i]
            print("generating", science_name)
            run("hotpants", template_name, science_name, conv_name, sub_name, RES_PATH, TEST_PATH)
        
    success = run_tests(verbose, tests, build_config, external_path)

    color_print.destroy()

    return success

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
