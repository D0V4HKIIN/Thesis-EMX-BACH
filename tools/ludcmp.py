J_COUNT = 9

for j in range(1, J_COUNT):
    print(f"j = {j}")

    for i in range(1, j):
        print(f"m{i}{j} = m{i}{j}", end="")

        for k in range(1, i):
            print(f" - m{i}{k} * m{k}{j}", end="")

        print()
