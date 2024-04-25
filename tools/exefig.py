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

COMPUTERS = [
    "gpu",
    "igpu"
]

def make_time(db, part, out_path):
    LABELS = [f"T{i}" for i in range(1, 13)]

    db = list(filter(lambda x: x[3] == part, db))

    if len(db) == 0:
        print(f"{color_print.RED}Invalid part specified, or missing execution time result.")
        return
    
    BIG_FONT = 16
    SMALL_FONT = 12

    fig, axes = plt.subplots(nrows=2)
    plt.rc("font", size=BIG_FONT)
    plt.subplots_adjust(left=0.14, bottom=0.04, right=0.96, top=0.94, hspace=0.26)
    #plt.yscale("log")
    fig.set_size_inches(8, 9, forward=True)

    for tmp in db:
        print(tmp)

    label_locations = np.arange(len(LABELS))
    PADDING = 0

    for i in range(len(axes)):
        ax = axes[i]
        computer = COMPUTERS[i]

        fig_db = list(filter(lambda x: x[0] == computer, db))

        data = [
            ("xbach", []),
            ("bach", []),
            ("hotpants", []),
            ]
        
        j = 0

        for software in ["xbach", "bach", "hotpants"]:
            for d in filter(lambda x: x[1] == software, fig_db):
                data[j][1].append(d[4])

            if len(data[j][1]) > 0:
                j += 1
            else:
                del data[j]

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
        multiplier = 0

        for (attribute, measurements) in data:
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
        ax.set_title(f"{part} execution time on computer {COMPUTER[computer]}")
        ax.set_xticks(label_locations + bar_width, labels=LABELS, fontsize=12)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylabel("Time [ms]", fontsize=BIG_FONT)
        ax.tick_params(labelsize=SMALL_FONT)

    fig.savefig(out_path / f"exec_{part.lower()}.pdf")

def make_speedup(db, part, out_path):
    LABELS = [f"T{i}" for i in range(1, 13)]

    db = list(filter(lambda x: x[3] == part, db))

    if len(db) == 0:
        print(f"{color_print.RED}Invalid part specified, or missing execution time result.")
        return
    
    BIG_FONT = 16
    SMALL_FONT = 12

    fig, axes = plt.subplots(nrows=2)
    plt.rc("font", size=BIG_FONT)
    plt.subplots_adjust(left=0.14, bottom=0.04, right=0.96, top=0.94, hspace=0.26)
    #plt.yscale("log")
    fig.set_size_inches(8, 9, forward=True)

    for tmp in db:
        print(tmp)

    label_locations = np.arange(len(LABELS))
    PADDING = 0

    for i in range(len(axes)):
        ax = axes[i]
        computer = COMPUTERS[i]

        fig_db = list(filter(lambda x: x[0] == computer, db))

        data = [
            ("xbach", []),
            ("bach", []),
            ("hotpants", []),
            ]
        
        j = 0

        for software in ["xbach", "bach", "hotpants"]:
            for d in filter(lambda x: x[1] == software, fig_db):
                data[j][1].append(statistics.median(d[4]))

            if len(data[j][1]) > 0:
                j += 1
            else:
                del data[j]

        for j in range(1, len(data)):
            for k in range(len(data[j][1])):
                data[j][1][k] /= data[0][1][k]

        if len(data) > 0:
            del data[0]

        bar_width = 0.29

        if len(data) == 1:
            bar_width = 0.5
        elif len(data) == 2:
            bar_width = 0.4
            
        base_offset = bar_width if len(data) % 2 != 0 else bar_width / 2
        multiplier = 0

        for (attribute, measurement) in data:
            # Base line, no speed-up
            ax.axhline(1, linestyle="--", linewidth=0.8, color="0.2")

            # Bar
            offset = base_offset + (bar_width + PADDING) * multiplier
            rects = ax.bar(label_locations + offset, measurement, bar_width, label=SOFTWARE[attribute], edgecolor="0.2", linewidth=0.5)
            #ax.bar_label(rects, padding=3)

            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.yaxis.grid(True)
        ax.set_title(f"{part} speed-up on computer {COMPUTER[computer]}")
        ax.set_xticks(label_locations + bar_width, labels=LABELS, fontsize=12)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylabel("X-BACH Speed-up", fontsize=BIG_FONT)
        ax.tick_params(labelsize=SMALL_FONT)

    fig.savefig(out_path / f"speedup_{part.lower()}.pdf")

def main(args):
    color_print.init()

    if len(args) != 2:
        print(f"{color_print.YELLOW}Usage: {pathlib.Path(__file__).name} <res-path> <out-path>")
        return

    res_path = pathlib.Path(args[0])
    out_path = pathlib.Path(args[1])

    db = timedb.load(res_path)

    for part in ["Total", "Ini", "SSS", "CMV", "CD", "KSC", "Conv", "Sub", "Fin"]:
        print(f"{color_print.CYAN}Making figure for {part} execution time")
        make_time(db, part, out_path)

        print(f"{color_print.CYAN}Making figure for {part} speed-up")
        make_speedup(db, part, out_path)

    color_print.destroy()

if __name__ == "__main__":
    main(sys.argv[1:])
