import multiprocessing
import pathlib
import numpy as np
from astropy.io import fits

COMPUTER = "lab"
NUM_CORES = multiprocessing.cpu_count() // 2
print("verify that", NUM_CORES, "is the right amount of cores!")

ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.resolve()
BIN_PATH = ROOT_PATH / "bin"
RES_PATH = ROOT_PATH / "res"
TEST_PATH = ROOT_PATH / "tests"
OUTPUT_PATH = TEST_PATH / "measurements"
CONFIG_PATH = ROOT_PATH / "tools" / "test_config.txt"

# read external path
EXTERNAL_PATH = None

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as input:
        path_str = input.readline().strip()
        EXTERNAL_PATH = pathlib.Path(path_str)

    print(f"Using external path: {EXTERNAL_PATH}")

# template image, science image
TEST_CASES = [
    ("test0", "test1"),
    ("testScience", "testTemplate"),
    # ("ptf_m82_s_2k", "ptf_m82_t_2k"),
    # ("sparse0", "sparse1"),
    # ("ztf_m1_s_3k", "ztf_m1_t_3k"),
    # ("skyM-T-4k", "skyM-S-4k"),
    # ("skyM-T-5k", "skyM-S-5k"),
    # ("skyM-T-6k", "skyM-S-6k"),
    # ("skyM-T-7k", "skyM-S-7k"),
    # ("skyM-T-8k", "skyM-S-8k"),
    # ("skyM-T-9k", "skyM-S-9k"),
    # ("skyM-T-10k", "skyM-S-10k")
]

TEST_INDEXES = [i + 1 for i in range(len(TEST_CASES))]
CORE_INDEXES = [i + 1 for i in range(NUM_CORES)]

PIXEL_PER_IMAGE = [
    img[0] * img[1]
    for img in [
        (
            fits.open(RES_PATH / f"{fits_name[0]}.fits").info(0)[0][5]
            if (RES_PATH / f"{fits_name[0]}.fits").exists()
            else fits.open(EXTERNAL_PATH / f"{fits_name[0]}.fits").info(0)[0][5]
        )
        for fits_name in TEST_CASES
    ]
]
print(PIXEL_PER_IMAGE)
# PIXEL_PER_IMAGE = [330**2, 1912 * 2025]

# cpu part, accelerator part
OPTIMAL_PART = [
    (0.0, 0.0),
    (0.04, 0.19),
    (0.16, 0.23),
    (0.04, 0.23),
    (0.16, 0.23),
    (0.16, 0.23),
    (0.16, 0.23),
]

TESTS_PER_ARGUMENT = 10
ARG_MIN = 0.0
ARG_MAX = 0.35
ARGS_TO_TEST = np.linspace(ARG_MIN, ARG_MAX, TESTS_PER_ARGUMENT)

# SOFTWARES = ["emxbach"]
SOFTWARES = ["bach", "xbach", "emxbach", "hotpants"]
ACCELERATOR_PLATFORM = 1
ACCELERATOR_DEVICE = 0

LABELS = [
    "Ini",
    "SSS",
    "CMV",
    "CD",
    "KSC",
    "kernel creation",
    "convolution",
    "Conv",
    "Sub",
    "Fin",
    "Total",
]


def test_filename(test):
    return f'{test["computer"]}-{test["software"]}-cores{test["core"]}-t{test["test"]}-cp{test["cpu"]:.2f}-ac{test["accelerator"]:.2f}'


def get_times(tests: list[tuple], res_path):
    db = []

    for test in tests:
        filename = test_filename(test) + ".txt"
        with open(res_path / filename, "r") as input:
            for line in input:
                if not line:
                    continue

                split = line.split(":")
                assert len(split) == 2

                label = split[0].strip()
                times_str = split[1].strip()

                time_split = times_str.split(" ")
                times = list(map(int, time_split))

                db.append(dict(test, label=label, times=times))

    return db


# tests to find the optimal cpu and accelerator part
def generate_arg_tests():
    db = []
    software = "emxbach"
    for test_id in TEST_INDEXES:
        for cpuPart in ARGS_TO_TEST:
            for acceleratorPart in ARGS_TO_TEST:
                db.append(
                    {
                        "computer": COMPUTER,
                        "software": software,
                        "test": test_id,
                        "core": NUM_CORES,
                        "cpu": cpuPart,
                        "accelerator": acceleratorPart,
                    }
                )

    return db


# uses the optimal cpu and accelerator part to test softwares
def generate_tests():
    db = []

    for software in SOFTWARES:
        for test_id in TEST_INDEXES:
            num_cores = 1
            if software == "emxbach":
                num_cores = NUM_CORES
            for core in range(1, num_cores + 1):
                # there is no benefit at running with high cpu part when testing with lower number of cores
                cpu_part = 0.0 if core != NUM_CORES else OPTIMAL_PART[test_id - 1][0]

                db.append(
                    {
                        "computer": COMPUTER,
                        "software": software,
                        "test": test_id,
                        "core": core,
                        "cpu": cpu_part,
                        "accelerator": OPTIMAL_PART[test_id - 1][1],
                    }
                )

    return db


def get_tests_from_args(args):
    tests = []

    for arg in args:
        if arg == "--args":
            tests.extend(generate_arg_tests())
        elif arg == "--normal":
            tests.extend(generate_tests())

    if len(tests) == 0:
        print("use --normal and --arg to generate tests")

    return tests
