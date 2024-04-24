import color_print
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import statistics
import sys
import timedb

SOFTWARE = {
    "xbach": "X-BACH",
    "bach": "BACH",
    "hotpants": "HOTPANTS"
}

COMPUTER = {
    "gpu": "A",
    "igpu": "B"
}

def main(args):
    color_print.init()

    if len(args) != 1:
        print(f"{color_print.YELLOW}Usage: {pathlib.Path(__file__).name} <res-path>")
        return

    res_path = pathlib.Path(args[0])

    db = timedb.load(res_path)

    PART = "Total"
    LABELS = [f"t{i}" for i in range(1, 13)]

    db = list(filter(lambda x: x[3] == PART, db))

    fig, axes = plt.subplots()

    for tmp in db:
        print(tmp)

    label_locations = np.arange(len(LABELS))

    for ax in [axes]:
        multiplier = 0

        data = [
            ("xbach", []),
            ("bach", []),
            ("hotpants", []),
            ]
        
        i = 0

        for software in ["xbach", "bach", "hotpants"]:
            for d in filter(lambda x: x[1] == software, db):
                data[i][1].append(statistics.median(d[4]))

            if len(data[i][1]) > 0:
                i += 1
            else:
                del data[i]

        bar_width = 0.3

        if len(data) == 2:
            bar_width = 0.4

        for (attribute, measurement) in data:
            offset = bar_width * multiplier
            rects = ax.bar(label_locations + offset, measurement, bar_width, label=SOFTWARE[attribute])
            #ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.yaxis.grid(True)
        ax.set_title(f"{PART} execution time on computer TODO")
        ax.set_xticks(label_locations + bar_width, labels=LABELS)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylabel("Time [ms]")

    plt.show()

    color_print.destroy()

if __name__ == "__main__":
    main(sys.argv[1:])
