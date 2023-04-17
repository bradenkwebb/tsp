#!/usr/bin/python3

import Drivers
import math
import random
from time import ctime
from multiprocessing import Manager, Process

# File name to save the table as
TABLE_FILE = "table.csv"
# Number of trials to average for each cell's value
NUM_TRIALS = 5
# Timeout given to each algorithm
TIMEOUT = 600

if __name__ == "__main__":
    # Write header of CSV
    with open(TABLE_FILE, "w") as output:
        output.write(",Random,,Greedy,,,Branch & Bound,,,Fancy,,\n")
        output.write("# Cities,"
            "Time (sec),Path Length,"
            "Time (sec),Path Length,% of Random,"
            "Time (sec),Path Length,% of Greedy,"
            "Time (sec),Path Length,% of Greedy\n")

    # Required number of cities to include in table
    city_num_list = [15, 30, 60, 100, 200]

    # Add additional numbers of cities to table
    # city_num_list += random.sample(list(range(3, 15)), 4)
    # city_num_list += random.sample(list(range(16, 30)), 4)
    # city_num_list += random.sample(list(range(31, 60)), 4)
    # city_num_list += random.sample(list(range(61, 100)), 4)
    # city_num_list += random.sample(list(range(101, 200)), 4)

    # Sort list to start with the smallest number of cities and work our way up
    city_num_list.sort()

    print(f"Number of cities list to calculate: {city_num_list}")

    # Create each row of table
    for idx, num_cities in enumerate(city_num_list):
        # Random seeds to use across all algorithms when evaluating this num_cities
        rand_seeds = random.sample(range(0, 1000), NUM_TRIALS)

        print()
        print(f"{ctime()}: Now calculating {num_cities} cities ({idx + 1}/{len(city_num_list)}) "
            f"with seeds {rand_seeds}")

        drivers = {
            "random": Drivers.randomDriver,
            "greedy": Drivers.greedyDriver,
            "bb": Drivers.branchAndBoundDriver,
            "fancy": Drivers.fancyDriver}

        results = {name: [] for name in drivers.keys()}

        # Use Manager to share result dictionaries across processes
        with Manager() as manager:
            for name, driver in drivers.items():
                print(f"{ctime()}: Processing {name}....")

                # List of results from the driver
                result_list = [manager.dict() for seed in rand_seeds]

                # Create initial processes
                # Elements in format of (Process, seed, manager_dictionary)
                process_list = [[Process(
                            target=driver,
                            args=(seed, num_cities, TIMEOUT),
                            kwargs={"multiprocessing_results": dictionary}),
                        seed,
                        dictionary]
                    for seed, dictionary in zip(rand_seeds, result_list)]

                # While there are still processes to be evaluated. In a while
                # loop because some might be killed off by the system
                while process_list:
                    # Start all processes
                    for process_info in process_list:
                        process_info[0].start()
                    # Wait for all processes to finish
                    for process_info in process_list:
                        process_info[0].join()

                    # If every single result dictionary is empty, that means there isn't enough compute
                    # space/power to run one instance of the driver even once. Try reducing TIMEOUT
                    if all([process_info[2].keys() == [] for process_info in process_list]):
                        print(f"{ctime()} Not even one run of {name} could finish in {TIMEOUT} seconds!")
                        print(f"{ctime()} Reducing TIMEOUT to {max(TIMEOUT - 60, 5)} seconds and retrying....")
                        TIMEOUT = max(TIMEOUT - 60, 5)

                    # Add each result that actually completed before the timeout to results
                    for process_pair_idx in range(len(process_list) - 1, -1, -1):
                        if (result := process_list[process_pair_idx][2]).keys():
                            if result.get("cost") != math.inf:
                                results[name].append((result.get("time"), result.get("cost")))
                            # If the process finished/timedout, pop it
                            process_list.pop(process_pair_idx)
                        else:
                            process_list[process_pair_idx][0] = Process(
                                target=driver,
                                args=(process_list[process_pair_idx][1], num_cities, TIMEOUT),
                                kwargs={"multiprocessing_results": process_list[process_pair_idx][2]})
                            print(f"{ctime()}: Recreating killed {name} process: "
                                f"seed {process_list[process_pair_idx][1]}")

        # Calculate average random numbers
        if results["random"]:
            random_time = round(
                sum([result[0] for result in results["random"]]) / len(results["random"]), 2)
            random_path = round(
                sum([result[1] for result in results["random"]]) / len(results["random"]))
        else:
            random_time = "TB"
            random_path = "TB"

        # Calculate average greedy numbers
        if results["greedy"]:
            greedy_time = round(
                sum([result[0] for result in results["greedy"]]) / len(results["greedy"]), 2)
            greedy_path = round(
                sum([result[1] for result in results["greedy"]]) / len(results["greedy"]))
            # Make greedy_pct "NA" if random_path was never found
            if random_path != "TB":
                greedy_pct = round(greedy_path / random_path, 2)
            else:
                greedy_pct = "NA"
        else:
            greedy_time = "TB"
            greedy_path = "TB"
            greedy_pct = "TB"

        # Calculate average branch and bound numbers. Only supposed to use results if majority didn't
        # timeout
        if len(results["bb"]) >= math.ceil(NUM_TRIALS / 2):
            branch_bound_time = round(
                sum([result[0] for result in results["bb"]]) / len(results["bb"]), 2)
            branch_bound_path = round(
                sum([result[1] for result in results["bb"]]) / len(results["bb"]))
            # Make branch_bound_pct "NA" if greedy_path was never found
            if greedy_path != "TB":
                branch_bound_pct = round(branch_bound_path / greedy_path, 2)
            else:
                branch_bound_pct = "NA"
        else:
            branch_bound_time = "TB"
            branch_bound_path = "TB"
            branch_bound_pct = "TB"

        # Calculate average fancy numbers. Only supposed to use results if majority didn't
        # timeout
        if len(results["fancy"]) >= math.ceil(NUM_TRIALS / 2):
            fancy_time = round(
                sum([result[0] for result in results["fancy"]]) / len(results["fancy"]), 2)
            fancy_path = round(
                sum([result[1] for result in results["fancy"]]) / len(results["fancy"]))
            # Make fancy_pct "NA" if greedy_path was never found
            if greedy_path != "TB":
                fancy_pct = round(fancy_path / greedy_path, 2)
            else:
                fancy_pct = "NA"
        else:
            fancy_time = "TB"
            fancy_path = "TB"
            fancy_pct = "TB"

        # Write row data to table
        with open(TABLE_FILE, "a") as output:
            output.write(",".join([str(x) for x in [
                num_cities,
                random_time,
                random_path,
                greedy_time,
                greedy_path,
                greedy_pct,
                branch_bound_time,
                branch_bound_path,
                branch_bound_pct,
                fancy_time,
                fancy_path,
                fancy_pct]]) + "\n")
