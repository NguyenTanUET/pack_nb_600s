#!/usr/bin/env python3
"""
Test script to check if makespan = 58 is feasible for Pack045.data
"""
from docplex.cp.model import *
import time
from pathlib import Path


def test_makespan_58(data_file, time_limit=1800):
    """
    Test if makespan = 58 is feasible for Pack045.data
    Uses extended time limit to thoroughly test feasibility
    """
    print(f"Testing makespan = 58 for {data_file}")
    print(f"Time limit: {time_limit} seconds")
    print("=" * 60)

    start_time = time.time()

    try:
        # Read data file
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])

            print(f"Problem size: {NB_TASKS} tasks, {NB_RESOURCES} resources")

            CAPACITIES = [int(v) for v in file.readline().split()]
            print(f"Resource capacities: {CAPACITIES}")

            TASKS = [[int(v) for v in file.readline().split()] for i in range(NB_TASKS)]

        # Extract data
        DURATIONS = [TASKS[t][0] for t in range(NB_TASKS)]
        DEMANDS = [TASKS[t][1:NB_RESOURCES + 1] for t in range(NB_TASKS)]
        SUCCESSORS = [TASKS[t][NB_RESOURCES + 2:] for t in range(NB_TASKS)]

        print(f"Task durations: {DURATIONS}")
        print(f"Sum of all durations: {sum(DURATIONS)}")

        # Create CP model
        mdl = CpoModel()

        # Create interval variables for tasks
        tasks = [interval_var(name=f'T{i + 1}', size=DURATIONS[i]) for i in range(NB_TASKS)]

        # Add precedence constraints
        precedence_count = 0
        for t in range(NB_TASKS):
            for s in SUCCESSORS[t]:
                if s > 0:  # Valid successor
                    mdl.add(end_before_start(tasks[t], tasks[s - 1]))
                    precedence_count += 1

        print(f"Added {precedence_count} precedence constraints")

        # Add resource capacity constraints
        resource_constraints = 0
        for r in range(NB_RESOURCES):
            resource_usage = [pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0]
            if resource_usage:
                mdl.add(sum(resource_usage) <= CAPACITIES[r])
                resource_constraints += 1
                print(f"Resource {r + 1}: {len(resource_usage)} tasks using this resource")

        print(f"Added {resource_constraints} resource constraints")

        # Create makespan and add constraint for 58
        makespan = max(end_of(t) for t in tasks)
        mdl.add(makespan <= 58)

        print("\nTesting makespan <= 58...")
        print("Starting solver...")

        # Solve with extended time and verbose output
        res = mdl.solve(
            TimeLimit=time_limit,
            LogVerbosity="Normal",  # Changed to Normal for detailed output
            SearchType="DepthFirst",  # Try depth-first search
            Workers=1  # Use single worker for consistency
        )

        solve_time = time.time() - start_time

        print(f"\nSolver finished in {solve_time:.2f} seconds")

        # FIX: Kiểm tra đúng cách
        if res is not None and res.is_solution():
            print("✅ RESULT: Makespan = 58 is FEASIBLE!")
            print(f"Actual makespan found: {res.get_value(makespan)}")

            # Print task schedule
            print("\nTask Schedule:")
            for i, task in enumerate(tasks):
                start_time_val = res.get_value(start_of(task))
                end_time_val = res.get_value(end_of(task))
                print(f"Task {i + 1}: Start={start_time_val}, End={end_time_val}, Duration={DURATIONS[i]}")

            return True, res.get_value(makespan), solve_time
        else:
            print("❌ RESULT: Makespan = 58 is INFEASIBLE!")
            print("No solution found within time limit")
            return False, None, solve_time

    except Exception as e:
        solve_time = time.time() - start_time
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, solve_time


def test_without_makespan_constraint(data_file, time_limit=1800):
    """
    Solve the same problem without makespan constraint to find optimal makespan
    """
    print(f"\nTesting without makespan constraint for {data_file}")
    print(f"Time limit: {time_limit} seconds")
    print("=" * 60)

    start_time = time.time()

    try:
        # Read data file (same as above)
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])

            CAPACITIES = [int(v) for v in file.readline().split()]
            TASKS = [[int(v) for v in file.readline().split()] for i in range(NB_TASKS)]

        # Extract data
        DURATIONS = [TASKS[t][0] for t in range(NB_TASKS)]
        DEMANDS = [TASKS[t][1:NB_RESOURCES + 1] for t in range(NB_TASKS)]
        SUCCESSORS = [TASKS[t][NB_RESOURCES + 2:] for t in range(NB_TASKS)]

        # Create CP model
        mdl = CpoModel()

        # Create interval variables for tasks
        tasks = [interval_var(name=f'T{i + 1}', size=DURATIONS[i]) for i in range(NB_TASKS)]

        # Add precedence constraints
        for t in range(NB_TASKS):
            for s in SUCCESSORS[t]:
                if s > 0:  # Valid successor
                    mdl.add(end_before_start(tasks[t], tasks[s - 1]))

        # Add resource capacity constraints
        for r in range(NB_RESOURCES):
            resource_usage = [pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0]
            if resource_usage:
                mdl.add(sum(resource_usage) <= CAPACITIES[r])

        # Create makespan but DON'T add constraint - instead minimize it
        makespan = max(end_of(t) for t in tasks)
        mdl.minimize(makespan)  # Minimize makespan instead of constraining it

        print("Minimizing makespan without upper bound constraint...")
        print("Starting solver...")

        # Solve with extended time
        res = mdl.solve(
            TimeLimit=time_limit,
            LogVerbosity="Normal",
            Workers=1
        )

        solve_time = time.time() - start_time

        print(f"\nSolver finished in {solve_time:.2f} seconds")

        # FIX: Kiểm tra đúng cách và sử dụng objective value
        if res is not None and res.is_solution():
            optimal_makespan = res.get_objective_value()  # Dùng get_objective_value() thay vì get_value(makespan)
            print(f"✅ RESULT: Optimal makespan = {optimal_makespan}")

            return True, optimal_makespan, solve_time
        else:
            print("❌ RESULT: No solution found")
            return False, None, solve_time

    except Exception as e:
        solve_time = time.time() - start_time
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, solve_time

def main():
    # Test file
    data_file = Path("data/Pack045.data")

    if not data_file.exists():
        print(f"Error: File {data_file} not found!")
        return

    print("RCPSP Makespan Test for Pack045.data")
    print("=" * 80)

    # Test 1: Check if makespan = 58 is feasible
    print("\n🔍 TEST 1: Testing if makespan = 58 is feasible")
    feasible_58, actual_makespan, time_58 = test_makespan_58(data_file, time_limit=1800)

    # Test 2: Find optimal makespan without constraint
    print("\n🔍 TEST 2: Finding optimal makespan without constraint")
    optimal_found, optimal_makespan, time_optimal = test_without_makespan_constraint(data_file, time_limit=1800)

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY OF RESULTS")
    print("=" * 80)

    print(f"Test 1 - Makespan = 58 feasible: {'YES' if feasible_58 else 'NO'}")
    if feasible_58 and actual_makespan:
        print(f"  Actual makespan found: {actual_makespan}")
    print(f"  Time taken: {time_58:.2f} seconds")

    print(f"\nTest 2 - Optimal makespan: {optimal_makespan if optimal_found else 'NOT FOUND'}")
    print(f"  Time taken: {time_optimal:.2f} seconds")

    if feasible_58 and optimal_found:
        if optimal_makespan == 58:
            print("\n✅ CONCLUSION: Makespan = 58 is indeed optimal!")
        elif optimal_makespan > 58:
            print(f"\n⚠️  CONCLUSION: Something is wrong! Optimal ({optimal_makespan}) > 58")
        else:
            print(f"\n❌ CONCLUSION: Makespan = 58 is NOT optimal. True optimal = {optimal_makespan}")

    elif not feasible_58 and optimal_found:
        print(f"\n✅ CONCLUSION: Makespan = 58 is infeasible. True optimal = {optimal_makespan}")
        if optimal_makespan == 612:
            print("  This matches the expected result of 612!")

    else:
        print("\n❓ CONCLUSION: Unable to determine conclusively due to solver timeouts")


if __name__ == "__main__":
    main()