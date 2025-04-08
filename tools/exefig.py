import color_print
import exetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import statistics
import sys

SOFTWARE = {
    "emxbach": "EMX-BACH",
    "xbach": "X-BACH",
    "bach": "BACH",
    "hotpants": "HOTPANTS"
}

SOFTWARES = [
    "emxbach",
    "xbach",
    "bach",
    "hotpants",
]

COMPUTER = {
    "dev": "dev",
    # "gpu": "A",
    # "igpu": "B"
}

BAR_COLORS = [
    "r",
    "g",
    "b"
]

COMPUTERS = [
    "dev",
    # "gpu",
    # "igpu"
]

NUM_COMPUTERS = len(COMPUTERS)
NUM_TESTS = 2 + 1

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()

def load_db(res_path):
    db = []

    for gpu in COMPUTERS:
        for software in SOFTWARES:
            for test_id in range(1, NUM_TESTS):
                test = f"t{test_id}"
                file_name = f"{gpu}-{software}-{test}.txt"

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

                        db.append((gpu, software, test, label, times))

    return db

def make_time_ax(db, ax, computer, part, big_font, small_font):
    LABELS = [f"T{i}" for i in range(1, NUM_TESTS)]
    label_locations = np.arange(len(LABELS))
    PADDING = 0

    fig_db = list(filter(lambda x: x[0] == computer, db))

    data = [[] for i in range(len(SOFTWARE))]
        # ("xbach", []),
        # ("bach", []),
        # ("hotpants", []),
        # ]
    
    j = 0

    for software in SOFTWARES:
        for d in filter(lambda x: x[1] == software, fig_db):
            data[j].append(d[4])

        if len(data[j]) > 0:
            j += 1
        else:
            print(software)
            print(data[j])
            print("idk what is going on but this is probably wrong")
            del data[j]
    
    print(data)

    bar_width = 1/len(data) - 0.01
    print(bar_width)

    if len(data) == 2:
        bar_width = 0.4

    max_time = 0

    for d in data:
        for m in d:
            measurement = statistics.median(m)

            if measurement > max_time:
                max_time = measurement
        
    base_offset = 0 if len(data) % 2 != 0 else bar_width / 2
    multiplier = 0

    for i, measurements in enumerate(data):
        measurement = []

        for m in measurements:
            measurement.append(statistics.median(m))

        # Bar
        offset = base_offset + (bar_width + PADDING) * multiplier
        rects = ax.bar(label_locations + offset, measurement, bar_width, label=SOFTWARES[i], edgecolor="0.2", linewidth=0.5)
        #ax.bar_label(rects, padding=3)

        # Variation
        for j, m in enumerate(measurements):
            (ci_lower, ci_upper) = exetime.calc_ci(m)

            CI_BOUND = 1 * 1e-2

            if (max_time == 0):
                print("max_time == 0")
                continue
            if (ci_upper - ci_lower) / max_time < CI_BOUND:
                continue

            var_start = label_locations[j] + offset - bar_width / 3
            var_end = label_locations[j] + offset + bar_width / 3
            var_mid = (var_start + var_end) / 2

            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
            # https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
            CI_LINE_WIDTH = 1
            CI_COLOR = "0"

            # ax.plot([var_start, var_end], [ci_lower, ci_lower], color=CI_COLOR, linewidth=CI_LINE_WIDTH)
            # ax.plot([var_start, var_end], [ci_upper, ci_upper], color=CI_COLOR, linewidth=CI_LINE_WIDTH)
            # ax.plot([var_mid, var_mid], [ci_lower, ci_upper], color=CI_COLOR, linewidth=CI_LINE_WIDTH)
            #ax.fill_between([var_start, var_end], ci_lower, ci_upper, color="b", alpha=0.2)

        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.yaxis.grid(True)
    ax.set_title(f"{part} execution time on computer {COMPUTER[computer]}")
    ax.set_xticks(label_locations + bar_width, labels=LABELS, fontsize=12)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylabel("Time [ms]", fontsize=big_font)
    ax.tick_params(labelsize=small_font)

def make_time(db, part, out_path):
    db = list(filter(lambda x: x[3] == part, db))

    if len(db) == 0:
        print(f"{color_print.RED}Invalid part specified, or missing execution time result.")
        return
    
    BIG_FONT = 16
    SMALL_FONT = 12

    fig, axes = plt.subplots(nrows=NUM_COMPUTERS)
    if type(axes) != list:
        axes = [axes]
    plt.rc("font", size=BIG_FONT)
    plt.subplots_adjust(left=0.14, bottom=0.04, right=0.96, top=0.94, hspace=0.26)
    fig.set_size_inches(8, 9, forward=True)

    for tmp in db:
        print(tmp)

    for i in range(len(axes)):
        make_time_ax(db, axes[i], COMPUTERS[i], part, BIG_FONT, SMALL_FONT)        

    fig.savefig(out_path / f"exec_{part.lower()}.pdf")

def make_time_separate(db, part, out_path):
    db = list(filter(lambda x: x[3] == part, db))

    if len(db) == 0:
        print(f"{color_print.RED}Invalid part specified, or missing execution time result.")
        return

    for tmp in db:
        print(tmp)
    
    BIG_FONT = 16
    SMALL_FONT = 12

    for computer in COMPUTERS:
        fig, ax = plt.subplots()
        plt.rc("font", size=BIG_FONT)
        plt.subplots_adjust(left=0.14, bottom=0.08, right=0.97, top=0.91)
        fig.set_size_inches(8, 9 / 2, forward=True)

        make_time_ax(db, ax, computer, part, BIG_FONT, SMALL_FONT)        

        fig.savefig(out_path / f"presentation_exec_{part.lower()}_{computer.lower()}.svg")

def make_relative_speed(db, part, out_path):
    LABELS = [f"T{i}" for i in range(1, NUM_TESTS)]

    db = list(filter(lambda x: x[3] == part, db))

    if len(db) == 0:
        print(f"{color_print.RED}Invalid part specified, or missing execution time result.")
        return
    
    BIG_FONT = 16
    SMALL_FONT = 12

    fig, axes = plt.subplots(nrows=NUM_COMPUTERS)
    if type(axes) != list:
        axes = [axes]
    plt.rc("font", size=BIG_FONT)
    plt.subplots_adjust(left=0.14, bottom=0.04, right=0.96, top=0.94, hspace=0.26)
    fig.set_size_inches(8, 9, forward=True)

    for tmp in db:
        print(tmp)

    label_locations = np.arange(len(LABELS))
    PADDING = 0

    for i in range(len(axes)):
        ax = axes[i]
        computer = COMPUTERS[i]

        fig_db = list(filter(lambda x: x[0] == computer, db))

        data = [[] for i in range(len(SOFTWARES))]
        
        j = 0

        for software in SOFTWARES:
            for d in filter(lambda x: x[1] == software, fig_db):
                data[j].append(statistics.median(d[4]))

            if len(data[j]) > 0:
                j += 1
            else:
                print(software)
                print(data[j])
                print("idk what is going on but this is probably wrong")
                del data[j]

        for j in range(1, len(data)):
            for k in range(len(data[j])):
                speed_up = 0

                if min(data[0][k], data[j][k]) > 5:
                    speed_up = data[j][k] / data[0][k]
            
                data[j][k] = speed_up

        if len(data) > 0:
            del data[0]

        bar_width = 0.29

        if len(data) == 1:
            bar_width = 0.5
        elif len(data) == 2:
            bar_width = 0.4
            
        base_offset = bar_width if len(data) % 2 != 0 else bar_width / 2
        multiplier = 0

        for i, measurement in enumerate(data):
            # Base line, no speed-up
            ax.axhline(1, linestyle="--", linewidth=0.8, color="0.2")

            # Bar
            offset = base_offset + (bar_width + PADDING) * multiplier
            rects = ax.bar(label_locations + offset, measurement, bar_width, label=SOFTWARES[i], edgecolor="0.2", linewidth=0.5)
            #ax.bar_label(rects, padding=3)

            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.yaxis.grid(True)
        ax.set_title(f"{part} relative speed on computer {COMPUTER[computer]}")
        ax.set_xticks(label_locations + bar_width, labels=LABELS, fontsize=12)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylabel("Relative speed", fontsize=BIG_FONT)
        ax.tick_params(labelsize=SMALL_FONT)

    fig.savefig(out_path / f"speedup_{part.lower()}.pdf")

def make_breakdown(db, out_path):
    db = list(filter(lambda x: x[1] == "xbach" and x[3] != "Total", db))

    for i in range(len(db)):
        db[i] = (db[i][0], db[i][1], db[i][2], db[i][3], statistics.median(db[i][4]))
    
    BIG_FONT = 16
    SMALL_FONT = 12

    fig, axes = plt.subplots(nrows=NUM_COMPUTERS)
    if type(axes) != list:
        axes = [axes]
    plt.rc("font", size=BIG_FONT)
    plt.subplots_adjust(left=0.12, bottom=0.04, right=0.80, top=0.94, hspace=0.26)
    fig.set_size_inches(8, 9, forward=True)

    BAR_WIDTH = 0.75
    LABELS = [f"T{i}" for i in range(1, NUM_TESTS)]
    PARTS = ["Ini", "SSS", "CMV", "CD", "KSC", "Conv", "Sub", "Fin"]
    label_locations = np.arange(len(LABELS))
    
    for i in range(len(axes)):
        ax = axes[i]
        computer = COMPUTERS[i]

        fig_db = list(filter(lambda x: x[0] == computer, db))

        test_sums = dict()

        for tid in range(1, NUM_TESTS):
            tstr = f"t{tid}"
            sum = 0

            for d in filter(lambda x: x[2] == tstr, fig_db):
                sum += d[4]

            test_sums[tstr] = sum

        bottom = np.zeros(len(LABELS))

        for part in PARTS:
            data = list(filter(lambda x: x[3] == part, fig_db))
            
            values = [d[4] / test_sums[d[2]] * 100 for d in data]

            ax.bar(LABELS, values, BAR_WIDTH, label=part, bottom=bottom)

            for j in range(len(values)):
                bottom[j] += values[j]
            
        ax.set_title(f"Execution time breakdown on computer {COMPUTER[computer]}")
        ax.set_xticks(label_locations, labels=LABELS, fontsize=12)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_ylabel("Percentage [%]", fontsize=BIG_FONT)
        ax.tick_params(labelsize=SMALL_FONT)

    fig.savefig(out_path / f"exec_breakdown.pdf")

def main(args):
    color_print.init()

    if len(args) != 2:
        print(f"{color_print.YELLOW}Usage: {pathlib.Path(__file__).name} <res-path> <out-path>")
        return

    res_path = pathlib.Path(args[0])
    out_path = pathlib.Path(args[1])

    db = load_db(res_path)

    print(f"{color_print.CYAN}Making figure for breakdown")
    make_breakdown(db, out_path)

    for part in ["Total", "Ini", "SSS", "CMV", "CD", "KSC", "Conv", "Sub", "Fin"]:
        print(f"{color_print.CYAN}Making figure for {part} execution time")
        make_time(db, part, out_path)
        make_time_separate(db, part, out_path)

        print(f"{color_print.CYAN}Making figure for {part} relative speed")
        make_relative_speed(db, part, out_path)

    color_print.destroy()

if __name__ == "__main__":
    main(sys.argv[1:])
