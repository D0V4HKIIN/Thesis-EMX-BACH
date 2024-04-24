import os
import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()

def load(res_path):
    db = []

    for gpu in ["igpu"]: # TODO: GPU
        for software in ["xbach", "bach", "hotpants"]:
            for test_id in range(1, 13):
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

