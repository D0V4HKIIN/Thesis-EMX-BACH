import color_print
import exetime
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

BAR_COLORS = [
    "r",
    "g",
    "b"
]

def main(args):
    color_print.init()

    if len(args) != 2:
        print(f"{color_print.YELLOW}Usage: {pathlib.Path(__file__).name} <res-path> <part>")
        return

    res_path = pathlib.Path(args[0])
    part = args[1]

    db = timedb.load(res_path)

    LABELS = [f"T{i}" for i in range(1, 13)]

    db = list(filter(lambda x: x[3] == part, db))

    if len(db) == 0:
        print(f"{color_print.RED}Invalid part specified, or missing execution time result.")
        return

    fig, axes = plt.subplots()
    #plt.yscale("log")

    for tmp in db:
        print(tmp)

    label_locations = np.arange(len(LABELS))
    PADDING = 0

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
                data[i][1].append(d[4])

            if len(data[i][1]) > 0:
                i += 1
            else:
                del data[i]

        bar_width = 0.29

        if len(data) == 2:
            bar_width = 0.4

        max_time = 0

        for d in data:
            for m in d[1]:
                measurement = statistics.median(m)

                if measurement > max_time:
                    max_time = measurement
            
        base_offset = 0 if len(data) % 2 != 0 else bar_width / 2

        for i, (attribute, measurements) in enumerate(data):
            measurement = []

            for m in measurements:
                measurement.append(statistics.median(m))

            # Bar
            offset = base_offset + (bar_width + PADDING) * multiplier
            rects = ax.bar(label_locations + offset, measurement, bar_width, label=SOFTWARE[attribute], edgecolor="0.2", linewidth=0.5)
            #ax.bar_label(rects, padding=3)

            # Variation
            for j, m in enumerate(measurements):
                (ci_lower, ci_upper) = exetime.calc_ci(m)

                CI_BOUND = 0.4 * 1e-2

                if (ci_upper - ci_lower) / max_time < CI_BOUND:
                    continue

                print(f"{ci_lower} {ci_upper} {max_time} {(ci_upper - ci_lower) / max_time}")

                var_start = label_locations[j] + offset - bar_width / 3
                var_end = label_locations[j] + offset + bar_width / 3
                var_mid = (var_start + var_end) / 2

                # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
                # https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
                CI_LINE_WIDTH = 1
                CI_COLOR = "0"

                ax.plot([var_start, var_end], [ci_lower, ci_lower], color=CI_COLOR, linewidth=CI_LINE_WIDTH)
                ax.plot([var_start, var_end], [ci_upper, ci_upper], color=CI_COLOR, linewidth=CI_LINE_WIDTH)
                ax.plot([var_mid, var_mid], [ci_lower, ci_upper], color=CI_COLOR, linewidth=CI_LINE_WIDTH)
                #ax.fill_between([var_start, var_end], ci_lower, ci_upper, color="b", alpha=0.2)

            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.yaxis.grid(True)
        ax.set_title(f"{part} execution time on computer TODO")
        ax.set_xticks(label_locations + bar_width, labels=LABELS)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylabel("Time [ms]")

    plt.show()

    color_print.destroy()

if __name__ == "__main__":
    main(sys.argv[1:])
