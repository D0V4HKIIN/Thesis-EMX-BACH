import color_print
import math
import statistics
import sys

def calc_ci(times):
    # For alpha = 0.05
    # From table which doesn't contain z(0.025), so use
    # (z(0.02) + z(0.03)) / 2
    z_alpha_2 = (0.50798 + 0.51197) / 2

    sorted_times = list(sorted(times))
    n = len(times)

    lower_rank = math.floor((n - z_alpha_2 * math.sqrt(n)) / 2)
    upper_rank = math.ceil(1 + (n + z_alpha_2 * math.sqrt(n)) / 2)

    lower = sorted_times[lower_rank]
    upper = sorted_times[upper_rank]

    return (lower, upper)

def main(args):
    e = 0.1

    color_print.init()

    times = list(map(int, args))
    n = len(times)

    print(f"Using times: {times}")
    print(f"n = {n}")

    if n < 6:
        print(f"{color_print.RED}Need at least 6 measurements!")
        return False
    
    (lower, upper) = calc_ci(times)
    median = statistics.median(times)

    print(f"Median  = {median}")    
    print(f"Rank v  = [{lower}, {upper}]")    

    allowed_min = (1 - e) * median
    allowed_max = (1 + e) * median

    print(f"Allowed = [{allowed_min}, {allowed_max}]")

    success = lower >= allowed_min and upper <= allowed_max

    if success:
        print(f"{color_print.GREEN}Enough measurements have been taken!")
    else:
        print(f"{color_print.RED}Needs more measurements!")

    color_print.destroy()

    return success

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
