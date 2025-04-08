import matplotlib.pyplot as plt
import sys
import pathlib
from  scipy import stats
import numpy as np

COMPUTERS = ["dev"]

CORE_PER_COMPUTER = [8]

SOFTWARES = ["bach", "xbach", "emxbach", "hotpants"]

NUM_TESTS = 2

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
        plt.hist(test["times"], bins=10)
        plt.show()

def graph_SSS_cores(db):
    fdb = list(filter(lambda x: x["software"] == "emxbach" and x["label"] == "SSS", db))
    print(fdb)

    for c, computer in enumerate(COMPUTERS):
        for t in range(1, NUM_TESTS + 1):
            n = CORE_PER_COMPUTER[c]
            means = np.zeros(n) # [None for i in range(n)]
            maxes = np.zeros(n) # [None for i in range(n)]
            mins = np.zeros(n) # [None for i in range(n)]
            tdb = list(filter(lambda x: x["test"] == t, fdb))

            for test in tdb:
                means[test["core"] - 1] = np.mean(test["times"])
                mins[test["core"] - 1] = np.min(test["times"])
                maxes[test["core"] - 1] = np.max(test["times"])
            
            print(means, maxes, mins)

            plt.title(f"{c} t{t} SSS")
            plt.xlabel("Cores")
            plt.ylabel("Execution time")
            plt.plot(range(n), means, marker="o")
            plt.fill_between(range(n), mins, maxes, color="red", alpha=0.2)
            plt.show()




    

def main(args):
    print(args[0])
    res_path = pathlib.Path(args[0])
    out_path = pathlib.Path(args[1])

    db = load_db(res_path)

    # verify_normality(db)

    # graph_tests(db)

    graph_SSS_cores(db)

    


if __name__ == "__main__":
    main(sys.argv[1:])
