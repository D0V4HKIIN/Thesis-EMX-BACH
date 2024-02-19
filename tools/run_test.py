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
    # ID | Fast? | Speed | Science       | Template      | HOTPANTS conv    | HOTPANTS sub    | Max abs error S,T | Max rel error S,T
    ( 1,   True,  "test0",        "test1",        "test01_conv",     "test01_sub",     (2e-4, 5e-4),       (5e-3, 4e-3)),
    ( 2,   True,  "testScience",  "testTemplate", "testST_conv",     "testST_sub",     (8e-3, 2e-3),       (5e-6, 9e-1)),
    ( 3,   False, "ptf_m82_s_2k", "ptf_m82_t_2k", "ptf_m82_2k_conv", "ptf_m82_2k_sub", (2e-1, 3e1),        (1e-5, 4e-1)),
    ( 4,   False, "sparse0",      "sparse1",      "sparse01_conv",   "sparse01_sub",   (2e1,  5e0),        (3e-4, 5e-4))
]

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
BUILD_PATH = ROOT_PATH / "build" / "Debug"
RES_PATH = ROOT_PATH / "res"
TEST_PATH = ROOT_PATH / "tests"
OUTPUT_PATH = TEST_PATH / "out"

def diff_fits(h_path, b_path):
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

def run_test(test_index, verbose):
    (id, _, science_name, template_name, conv_name, sub_name, max_abs_error, max_rel_error) = TEST_TABLE[test_index]

    print(f"{color_print.CYAN}Running test {id}...")

    exe_path = BUILD_PATH / "BACH.exe"

    exe_args = [str(exe_path)]
    exe_args += ["-ip", str(RES_PATH)]
    exe_args += ["-s", f"{science_name}.fits"]
    exe_args += ["-t", f"{template_name}.fits"]
    exe_args += ["-op", str(OUTPUT_PATH / f"test{id}_")]

    start_time = time.time()

    with open(OUTPUT_PATH / f"test{id}_out.txt", "w") as out_stream:
        if not subprocess.run(args=exe_args, stdout=out_stream, stderr=out_stream):
            print(f"{color_print.RED}X-BACH exited with an error code!")
            return False

    CONV_OUT_PATH = OUTPUT_PATH / f"test{id}_diff.fits"
    SUB_OUT_PATH = OUTPUT_PATH / f"test{id}_sub.fits"

    if not CONV_OUT_PATH.exists() or not SUB_OUT_PATH.exists():
        print("At least one X-BACH output is missing. The program did not run correctly.")
        return False

    end_time = time.time()
    test_time = end_time - start_time

    conv_max_abs_err, conv_mean_abs_err, conv_max_rel_err, conv_mean_rel_err, conv_wrong_nans, conv_max_coords = diff_fits(TEST_PATH / f"{conv_name}.fits", CONV_OUT_PATH)
    sub_max_abs_err, sub_mean_abs_err, sub_max_rel_err, sub_mean_rel_err, sub_wrong_nans, sub_max_coords = diff_fits(TEST_PATH / f"{sub_name}.fits", SUB_OUT_PATH)

    print(f"Test took {test_time:.2f} seconds")
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

def run_tests(verbose, tests):
    print(f"{color_print.CYAN}Running X-BACH from \"{BUILD_PATH.resolve()}\"")

    print()
    print(f"There are a total of {len(tests)} tests to run:")

    for i in tests:
        test = TEST_TABLE[i]
        print(f" Test {test[0]}, Science: {test[2]} Template: {test[3]}")

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

    for i in tests:
        test_id = TEST_TABLE[i][0]
        test_success = run_test(i, verbose)
        
        if test_success:
            print(f"{color_print.GREEN}Test {test_id} succeeded!")
        else:
            failed_tests += 1
            print(f"{color_print.RED}Test {test_id} failed!")

        total_tests += 1
        print()

    if failed_tests > 0:
        print(f"{color_print.RED}{failed_tests} / {total_tests} tests failed!")

        return False
    
    print(f"{color_print.GREEN}All tests were successful!")

def main(args):
    color_print.init()

    # Parse args
    verbose = False
    tests = {}

    for arg in args:
        if arg == "-h":
            print_help()
            return True
        elif arg == "-v":
            verbose = True
        elif arg == "--all":
            tests = [i for i in range(len(TEST_TABLE))]
        elif arg == "--fast":
            tests = [i for i in range(len(TEST_TABLE)) if TEST_TABLE[i][1]]
        elif arg == "--slow":
            tests = [i for i in range(len(TEST_TABLE)) if not TEST_TABLE[i][1]]
        else:
            print(f"{color_print.RED}Unrecognized flag: {arg}")
            print()
            print_help()
            return False

    if len(tests) == 0:
        tests = [i for i in range(len(TEST_TABLE))]

    success = run_tests(verbose, tests)

    color_print.destroy()

    return success

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
