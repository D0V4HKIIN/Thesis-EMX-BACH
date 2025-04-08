import matplotlib.pyplot as plt
import sys
import pathlib
from  scipy import stats
import numpy as np

COMPUTERS = ["dev"]

CORE_PER_COMPUTER = [8]

OPTIMAL_CORE_PER_COMPUTER = [4]

SOFTWARES = ["bach", "xbach", "emxbach", "hotpants"]

NUM_TESTS = 2

PixelPerImage = [330**2, 1912 * 2025]

def load_db(res_path):
    db = []

    for i, computer in enumerate(COMPUTERS):
        for software in SOFTWARES:
            num_cores = 1
            if software == "emxbach":
                num_cores = CORE_PER_COMPUTER[i]
            for core in range(1, num_cores + 1):
                for test_id in range(1, NUM_TESTS + 1):
                    test = f"t{test_id}"
                    file_name = f"{computer}-{software}-{core}-{test}.txt"

                    with open(res_path / file_name, "r") as input:
                        for line in input:
                            if not line:
                                continue

                            split = line.split(":")
                            assert len(split) == 2

                            label = split[0].strip()
                            times_str = split[1].strip()

                            time_split = times_str.split(" ")
                            times = list(map(int, time_split))

                            db.append({"computer":computer, "software":software, "test":test_id, "core":core, "label":label, "times":times})

    return db

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

def graph_cores(db, label):
    fdb = list(filter(lambda x: x["software"] == "emxbach" and x["label"] == label, db))

    for c, computer in enumerate(COMPUTERS):
        for t in range(1, NUM_TESTS + 1):
            tdb = list(filter(lambda x: x["test"] == t, fdb))
            n = CORE_PER_COMPUTER[c]
            means = np.zeros(n) # [None for i in range(n)]
            maxes = np.zeros(n) # [None for i in range(n)]
            mins = np.zeros(n) # [None for i in range(n)]

            for test in tdb:
                means[test["core"] - 1] = np.mean(test["times"])
                mins[test["core"] - 1] = np.min(test["times"])
                maxes[test["core"] - 1] = np.max(test["times"])
            
            plt.title(f"{computer} t{t} {label}")
            plt.xlabel("Cores")
            plt.ylabel("Execution time")
            plt.plot(range(1, n + 1), means, marker="o")
            plt.fill_between(range(1, n + 1), mins, maxes, color="red", alpha=0.2)
            plt.show()

def arrayify(dict):
    labels = dict.keys()
    values = []
    for label in labels:
        values.append(dict[label])
    return labels, values

def graph_diff(db, label):
    fdb = list(filter(lambda x: x["label"] == label, db))

    for c, computer in enumerate(COMPUTERS):
        for t in range(1, NUM_TESTS + 1):
            tdb = list(filter(lambda x: x["test"] == t and x["computer"] == computer and (x["software"] != "emxbach" or x["core"] == OPTIMAL_CORE_PER_COMPUTER[c]), fdb))
           
            means = {}
            maxes = {}
            mins = {}

            for test in tdb:
                id = (test["software"], test["core"])
                means[id] = np.mean(test["times"])
                maxes[id] = np.max(test["times"])
                mins[id] = np.min(test["times"])
            
            labels, values = arrayify(means)

            plt_labels = [l[0] + str(l[1]) for l in labels]

            # doesnt show min or max
            plt.figure(figsize=(15,7))
            plt.title(f"Comparison {computer} t{t} {label}")
            plt.xlabel("Software")
            plt.ylabel("Execution time")
            
            for key in means:
                plt.bar(key[0] + str(key[1]), means[key])

            plt.legend()
            plt.show()

def graph_total(db):
    label = "Total"
    fdb = list(filter(lambda x: x["label"] == label, db))

    for c, computer in enumerate(COMPUTERS):
        for software in SOFTWARES:
            means = {}
            maxes = {}
            mins = {}

            for t in range(1, NUM_TESTS + 1):
                tdb = list(filter(lambda x: x["test"] == t and x["computer"] == computer and (x["software"] != "emxbach" or x["core"] == OPTIMAL_CORE_PER_COMPUTER[c]), fdb))


                for test in tdb:
                    id = (test["software"], test["core"])
                    means[id] = means.get(id, []) + [np.mean(test["times"])]
                    maxes[id] = maxes.get(id, []) + [np.max(test["times"])]
                    mins[id] = mins.get(id, []) + [np.min(test["times"])]

        plt.title(f"Comparison {computer} {label}")
        plt.xlabel("Pixel Per Image")
        plt.ylabel("Execution time")

        for key in means:
            plt.plot(PixelPerImage, means[key], label=key[0] + str(key[1]))
            plt.fill_between(PixelPerImage, mins[key], maxes[key], alpha=0.2)
        
        plt.legend()
        plt.show()


def main(args):
    res_path = pathlib.Path(args[0])
    out_path = pathlib.Path(args[1])

    db = load_db(res_path)

    # verify_normality(db)

    # graph_tests(db)

    graph_cores(db, "SSS")
    graph_cores(db, "MakeKernels")
    graph_cores(db, "Convolution")

    graph_diff(db, "SSS")
    graph_diff(db, "MakeKernels")
    graph_diff(db, "Convolution")

    graph_total(db)
    


if __name__ == "__main__":
    main(sys.argv[1:])
