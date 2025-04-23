import matplotlib.pyplot as plt
import matplotlib
import sys
import pathlib
from scipy import stats
import numpy as np
import itertools
import os

from test_generator import *

CONFIDENCE_INTERVAL = 0.99

DPI = 300


def label_string(key):
    core_string = (
        f"\n{str(key[1])} core{"s" if key[1] != 1 else ""}"
        if key[0] == "emxbach"
        else ""
    )
    return f"{key[0]}{core_string}"


def verify_normality(db):
    for test in db:
        # test for normality using shapiro-wilk test
        shapiro = stats.shapiro(test["times"])

        if shapiro.pvalue < 0.05 or shapiro.statistic < 0.95:
            print("test not normal", test)
            print(shapiro)


def graph_tests(db):
    for test in db:
        plt.title(f"{test}")
        plt.hist(test["times"], bins=10)
        plt.show()


def graph_cores(db, label, out_path):
    fdb = list(filter(lambda x: x["software"] == "emxbach" and x["label"] == label, db))

    sub_folder = "cores"
    new_out_path = out_path / sub_folder
    if not new_out_path.exists():
        os.mkdir(out_path / sub_folder)

    for t in TEST_INDEXES:
        tdb = list(filter(lambda x: x["test"] == t, fdb))
        n = NUM_CORES
        means = np.zeros(n)
        low_percentile = np.zeros(n)
        high_percentile = np.zeros(n)

        for test in tdb:
            means[test["core"] - 1] = np.mean(test["times"])

            boot = stats.bootstrap(
                (test["times"],),  # for some reason this needs to be 2d
                np.mean,  # want to bootstrap the mean
                confidence_level=CONFIDENCE_INTERVAL,
                method="percentile",
            )
            low_percentile[test["core"] - 1] = boot.confidence_interval.low
            high_percentile[test["core"] - 1] = boot.confidence_interval.high

        title = f"Execution time of {label} for test {t}"
        plt.title(title)
        plt.xlabel("Cores")
        plt.ylabel("Execution time (ms)")
        plt.xticks(CORE_INDEXES)
        plt.plot(CORE_INDEXES, means, marker="o")
        plt.fill_between(
            CORE_INDEXES, low_percentile, high_percentile, color="red", alpha=0.2
        )

        # plt.show()

        filename = f"core_{label}_test{t}_computer{COMPUTER}"
        plt.savefig(new_out_path / filename, dpi=DPI)
        plt.close()


def graph_efficiency(db, label, out_path):
    fdb = list(filter(lambda x: x["software"] == "emxbach" and x["label"] == label, db))

    sub_folder = "cores"
    new_out_path = out_path / sub_folder
    if not new_out_path.exists():
        os.mkdir(out_path / sub_folder)

    for t in TEST_INDEXES:
        tdb = list(filter(lambda x: x["test"] == t, fdb))
        n = NUM_CORES
        means = np.zeros(n)
        low_percentile = np.zeros(n)
        high_percentile = np.zeros(n)

        for test in tdb:
            means[test["core"] - 1] = np.mean(test["times"])

            boot = stats.bootstrap(
                (test["times"],),  # for some reason this needs to be 2d
                np.mean,  # want to bootstrap the mean
                confidence_level=CONFIDENCE_INTERVAL,
                method="percentile",
            )
            low_percentile[test["core"] - 1] = boot.confidence_interval.low
            high_percentile[test["core"] - 1] = boot.confidence_interval.high

        one_mean = means[0]

        speedup_means = one_mean / means
        speedup_low = one_mean / low_percentile
        speedup_high = one_mean / high_percentile

        efficiency_means = speedup_means / CORE_INDEXES
        efficiency_low = speedup_low / CORE_INDEXES
        efficiency_high = speedup_high / CORE_INDEXES

        title = f"Efficiency of {label} for test {t}"
        plt.title(title)
        plt.xlabel("Cores")
        plt.ylabel("Efficiency")
        plt.xticks(CORE_INDEXES)
        plt.plot(CORE_INDEXES, efficiency_means, marker="o")
        plt.fill_between(
            CORE_INDEXES, efficiency_low, efficiency_high, color="red", alpha=0.2
        )

        # plt.show()

        filename = f"core_efficiency_{label}_test{t}_computer{COMPUTER}"
        plt.savefig(new_out_path / filename, dpi=DPI)
        plt.close()


def graph_together_efficiency(db, label, out_path):
    fdb = list(filter(lambda x: x["software"] == "emxbach" and x["label"] == label, db))

    sub_folder = "cores"
    new_out_path = out_path / sub_folder
    if not new_out_path.exists():
        os.mkdir(out_path / sub_folder)

    for t in TEST_INDEXES:
        tdb = list(filter(lambda x: x["test"] == t, fdb))
        n = NUM_CORES
        means = np.zeros(n)
        low_percentile = np.zeros(n)
        high_percentile = np.zeros(n)

        for test in tdb:
            means[test["core"] - 1] = np.mean(test["times"])

            boot = stats.bootstrap(
                (test["times"],),  # for some reason this needs to be 2d
                np.mean,  # want to bootstrap the mean
                confidence_level=CONFIDENCE_INTERVAL,
                method="percentile",
            )
            low_percentile[test["core"] - 1] = boot.confidence_interval.low
            high_percentile[test["core"] - 1] = boot.confidence_interval.high

        one_mean = means[0]

        speedup_means = one_mean / means
        speedup_low = one_mean / low_percentile
        speedup_high = one_mean / high_percentile

        efficiency_means = speedup_means / CORE_INDEXES
        efficiency_low = speedup_low / CORE_INDEXES
        efficiency_high = speedup_high / CORE_INDEXES

        plt.plot(CORE_INDEXES, efficiency_means, label=f"test {t}", marker="o")
        plt.fill_between(CORE_INDEXES, efficiency_low, efficiency_high, alpha=0.2)

    title = f"Efficiency of {label}"
    plt.title(title)
    plt.xlabel("Cores")
    plt.ylabel("Efficiency")
    plt.xticks(CORE_INDEXES)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.80, top=0.94, hspace=0.26)

    filename = f"core_togetherefficiency_{label}_computer{COMPUTER}"
    plt.savefig(new_out_path / filename, dpi=DPI)
    plt.close()


def graph_speedup(db, label, out_path):
    fdb = list(filter(lambda x: x["software"] == "emxbach" and x["label"] == label, db))

    sub_folder = "cores"
    new_out_path = out_path / sub_folder
    if not new_out_path.exists():
        os.mkdir(out_path / sub_folder)

    for t in TEST_INDEXES:
        tdb = list(filter(lambda x: x["test"] == t, fdb))
        n = NUM_CORES
        means = np.zeros(n)
        low_percentile = np.zeros(n)
        high_percentile = np.zeros(n)

        for test in tdb:
            means[test["core"] - 1] = np.mean(test["times"])

            boot = stats.bootstrap(
                (test["times"],),  # for some reason this needs to be 2d
                np.mean,  # want to bootstrap the mean
                confidence_level=CONFIDENCE_INTERVAL,
                method="percentile",
            )
            low_percentile[test["core"] - 1] = boot.confidence_interval.low
            high_percentile[test["core"] - 1] = boot.confidence_interval.high

        one_mean = means[0]

        speedup_means = one_mean / means
        speedup_low = one_mean / low_percentile
        speedup_high = one_mean / high_percentile

        title = f"Speedup of {label} for test {t}"
        plt.title(title)
        plt.xlabel("Cores")
        plt.ylabel("Speedup")
        plt.xticks(CORE_INDEXES)
        ax = plt.gca()
        ax.set_ylim(1, NUM_CORES)
        plt.plot(CORE_INDEXES, speedup_means, marker="o")
        plt.fill_between(
            CORE_INDEXES, speedup_low, speedup_high, color="red", alpha=0.2
        )

        # plt.show()

        filename = f"core_speedup_{label}_test{t}_computer{COMPUTER}"
        plt.savefig(new_out_path / filename, dpi=DPI)
        plt.close()


STEPS = [
    "Ini",
    "SSS",
    "CMV",
    "CD",
    "KSC",
    "Conv",
    "Sub",
    "Fin",
    "Total",
]


def graph_breakdown(db, software, out_path):
    num_cores = NUM_CORES if software == "emxbach" else 1
    fdb = list(
        filter(
            lambda x: x["software"] == software and x["core"] == num_cores,
            db,
        )
    )

    labels = [f"test {t}" for t in TEST_INDEXES]
    all_fractions = np.zeros((len(TEST_INDEXES), len(STEPS) - 1))

    for t in TEST_INDEXES:
        tdb = list(filter(lambda x: x["test"] == t, fdb))

        times = np.zeros(len(STEPS))

        for i, label in enumerate(STEPS):
            ldb = list(filter(lambda x: x["label"] == label, tdb))

            times[i] = np.sum(ldb[0]["times"])

        fractions = times[0:-1] / times[-1]  # last is total
        all_fractions[t - 1] = fractions

    bottoms = np.zeros(len(TEST_INDEXES))

    all_fractions *= 100  # in percentage

    for i in range(len(STEPS) - 1):
        fractions = all_fractions[:, i]
        plt.bar(labels, fractions, label=STEPS[i], bottom=bottoms)
        bottoms += fractions

    plt.title(f"Execution time breakdown of {software}")
    plt.ylabel("Percentage of total execution time")
    plt.subplots_adjust(left=0.12, bottom=0.08, right=0.80, top=0.94, hspace=0.26)
    plt.legend(bbox_to_anchor=(1, 1))

    # plt.show()

    filename = f"breakdown_{software}_computer{COMPUTER}"
    plt.savefig(out_path / filename, dpi=DPI)
    plt.close()


def arrayify(dict):
    labels = dict.keys()
    values = []
    for label in labels:
        values.append(dict[label])
    return labels, values


def graph_diff(db, label, out_path):
    fdb = list(filter(lambda x: x["label"] == label, db))

    sub_folder = "diff"
    new_out_path = out_path / sub_folder
    if not new_out_path.exists():
        os.mkdir(out_path / sub_folder)

    for t in TEST_INDEXES:
        tdb = list(
            filter(
                lambda x: x["test"] == t
                and x["computer"] == COMPUTER
                and (x["software"] != "emxbach" or x["core"] == NUM_CORES),
                fdb,
            )
        )

        means = {}
        low_percentile = {}
        high_percentile = {}

        for test in tdb:
            id = (test["software"], test["core"])
            means[id] = np.mean(test["times"])

            boot = stats.bootstrap(
                (test["times"],),  # for some reason this needs to be 2d
                np.mean,  # want to bootstrap the mean
                confidence_level=CONFIDENCE_INTERVAL,
                method="percentile",
            )
            low_percentile[id] = boot.confidence_interval.low
            high_percentile[id] = boot.confidence_interval.high

        title = f"Comparison of execution time of {label} for test {t}"
        # doesnt show min or max
        plt.figure(figsize=(5, 6))
        plt.title(title)
        plt.ylabel("Execution time (ms)")

        for key in means:
            plt_label = label_string(key)
            plt.bar(
                plt_label,
                means[key],
                yerr=[
                    [means[key] - low_percentile[key]],
                    [high_percentile[key] - means[key]],
                ],
            )

        # plt.show()

        filename = f"diff_{label}_test{t}_computer{COMPUTER}"
        plt.savefig(new_out_path / filename, dpi=DPI)
        plt.close()


def graph_total(db, label, out_path):
    fdb = list(filter(lambda x: x["label"] == label, db))

    for software in SOFTWARES:
        means = {}
        low_percentile = {}
        high_percentile = {}

        for t in TEST_INDEXES:
            tdb = list(
                filter(
                    lambda x: x["test"] == t
                    and x["computer"] == COMPUTER
                    and (x["software"] != "emxbach" or x["core"] == NUM_CORES),
                    fdb,
                )
            )

            for test in tdb:
                id = (test["software"], test["core"])
                means[id] = means.get(id, []) + [np.mean(test["times"])]

                boot = stats.bootstrap(
                    (test["times"],),  # for some reason this needs to be 2d
                    np.mean,  # want to bootstrap the mean
                    confidence_level=CONFIDENCE_INTERVAL,
                    method="percentile",
                )
                low_percentile[id] = low_percentile.get(id, []) + [
                    boot.confidence_interval.low
                ]
                high_percentile[id] = high_percentile.get(id, []) + [
                    boot.confidence_interval.high
                ]

    title = f"Comparison of execution time of {label}"
    plt.title(title)
    plt.xlabel("Pixel Per Image")
    plt.ylabel("Execution time (ms)")

    for key, marker in zip(means, itertools.cycle("os^d")):
        plt_label = label_string(key)
        plt.plot(PIXEL_PER_IMAGE, means[key], label=plt_label, marker=marker)
        plt.fill_between(
            PIXEL_PER_IMAGE, low_percentile[key], high_percentile[key], alpha=0.2
        )

    plt.legend()

    # plt.show()

    filename = f"line{label}_computer{COMPUTER}"
    plt.savefig(out_path / filename, dpi=DPI)
    plt.close()


def graph_args(db, out_path):
    fdb = list(filter(lambda x: x["label"] == "Convolution", db))

    for test in TEST_INDEXES:
        tdb = list(filter(lambda x: x["test"] == test, fdb))
        data = np.zeros(shape=(TESTS_PER_ARGUMENT, TESTS_PER_ARGUMENT))

        best_mean = np.inf
        best_cpu = np.inf
        best_acc = np.inf

        for m in tdb:
            mean = np.mean(m["times"])
            cpu_part = m["cpu"]
            acc_part = m["accelerator"]

            if mean < best_mean:
                best_mean = mean
                best_cpu = cpu_part
                best_acc = acc_part

            data[
                round(cpu_part * (TESTS_PER_ARGUMENT - 1) * 1 / (ARG_MAX - ARG_MIN)),
                round(acc_part * (TESTS_PER_ARGUMENT - 1) * 1 / (ARG_MAX - ARG_MIN)),
            ] = mean

        title = f"Effect of work distribution on execution time for test {test}"

        plt.suptitle(title)
        plt.title(
            f"Best execution time ({best_mean} ms) for cpu part = {best_cpu:.2f} and acc part = {best_acc:.2f}"
        )
        plt.xlabel("part of convolution done on accelerator")
        plt.ylabel("part of convolution done on cpu")
        plt.xticks(ARGS_TO_TEST, rotation=45)
        plt.yticks(ARGS_TO_TEST)

        # log_intensity = -1
        # logseq = np.logspace(log_intensity, 0)
        # cmap = plt.cm.rainbow(logseq)
        # cmap = plt.colormaps["viridis_r"]
        plt.pcolormesh(
            ARGS_TO_TEST,
            ARGS_TO_TEST,
            data,
            # this is bad
            # norm=matplotlib.colors.PowerNorm(0.5),
            cmap="viridis_r",
        )
        clb = plt.colorbar()
        clb.ax.set_ylabel("execution time in ms")

        plt.tight_layout()

        # plt.show()

        filename = f"arg_test{test}_computer{COMPUTER}"
        plt.savefig(out_path / filename, dpi=DPI)
        plt.close()


def main(args):
    res_path = pathlib.Path(args[0])
    out_path = pathlib.Path(args[1])

    print("res path:", res_path)
    print("out path:", out_path)

    plt.rcParams["figure.figsize"] = [6.8, 4]

    # verify_normality(db)

    # graph_tests(db)

    if "--normal" in args:
        test_db = get_times(generate_tests(), res_path)

        graph_cores(test_db, "SSS", out_path)
        graph_cores(test_db, "MakeKernels", out_path)

        graph_efficiency(test_db, "SSS", out_path)
        graph_efficiency(test_db, "MakeKernels", out_path)

        graph_together_efficiency(test_db, "SSS", out_path)
        graph_together_efficiency(test_db, "MakeKernels", out_path)

        graph_speedup(test_db, "SSS", out_path)
        graph_speedup(test_db, "MakeKernels", out_path)

        graph_diff(test_db, "SSS", out_path)
        graph_diff(test_db, "MakeKernels", out_path)
        graph_diff(test_db, "Convolution", out_path)
        graph_diff(test_db, "Conv", out_path)

        graph_total(test_db, "Total", out_path)
        graph_total(test_db, "SSS", out_path)
        graph_total(test_db, "MakeKernels", out_path)
        graph_total(test_db, "Convolution", out_path)
        graph_total(test_db, "Conv", out_path)

        graph_breakdown(test_db, "bach", out_path)
        graph_breakdown(test_db, "xbach", out_path)
        graph_breakdown(test_db, "emxbach", out_path)

    if "--args" in args:
        arg_db = get_times(generate_arg_tests(), res_path)

        graph_args(arg_db, out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
