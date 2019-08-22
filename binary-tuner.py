#==========================================#
# My own version of the Futhark Auto Tuner #
#==========================================#
from __future__ import print_function

import tempfile
import subprocess
import re
import shlex
import sys
import bisect
import json
import sys
import os
import math
import itertools
import time
import logging
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from collections import OrderedDict, defaultdict

# Simple way of calling a shell command, and wait for it to finish.
# Could maybe be extended if needed.
def call_program(cmd):
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
    retval = process.wait()

    return process

# Perform all the preparation from the original Auto Tuner
# Gets all values that thresholds are compared against (stored in "thresholds" and "values")
# Gets the names of all the datasets involved in tuning (stored in "datasets")
# Computes a "branch tree" of versions from threshold dependencies (stored in "branch_tree")
def extract_thresholds_and_values(program):
    # Call the --print-sizes to extract all threshold names.
    sizes_cmd = './{} --print-sizes'.format(program[:-4])
    result = call_program(sizes_cmd)

    # Use regex to find all the names, for each line in the output of the earlier call.
    all_params = []
    size_p = re.compile('([^ ]*) \(threshold+ \((.*?)\)\)')
    for line in result.stdout.readlines():
      m = size_p.search(line)
      if m:
        all_params.append(m.group(1))

    # Initialize the strutures for the results.
    thresholds = OrderedDict()
    datasets = []
    values = defaultdict(list)

    with tempfile.NamedTemporaryFile() as json_tmp:
        # extract comparison values by running a single run of the benchmark.
        # Saves important info to a temporary JSON file.
        val_cmd = 'futhark bench {} --exclude-case=notune --backend=opencl --skip-compilation --pass-option=-L --runs=1 --json={}'.format(
            program, json_tmp.name)

        val_res = call_program(val_cmd)

        json_data = json.load(json_tmp, object_pairs_hook=OrderedDict)

        # Datasets here contains all the runtimes for each dataset.
        datasets = json_data[program]['datasets']

        # search for parameters and values for each dataset
        val_re = re.compile('Compared ([^ ]+) <= (-?\d+)')
        for dataset in datasets:
            thresholds[dataset] = defaultdict(list)

            for line in datasets[dataset]['stderr'].splitlines():
                match = val_re.search(line)
                if match:
                    param, value = match.group(1), int(match.group(2))

                    # Add comparison; note there might be several per parameter.
                    if value not in thresholds[dataset][param]:
                        thresholds[dataset][param].append(value)

            # if a param has not been used in a comparison,
            # and thus not added, add it now
            for p in all_params:
                if not p in thresholds[dataset]:
                    thresholds[dataset][p] = []
                else:
                    params = sorted(thresholds[dataset][p])
                    thresholds[dataset][p] = params

        # we need a list of datasets later, so save just the names
        datasets = datasets.keys()


    # Make a simpler list of all the comparisons for each threshold.
    t = defaultdict(list)
    for dataset, thresh in thresholds.items():
        for k, vs in thresh.items():
            for v in vs:
                if v not in t[k]:
                    bisect.insort(t[k], v)


    # iterate over all sorted threshold values of all parameters
    # For each threshold-comparison:
    #Add 1 as a choice (Representing True choice)
    #For each comparison uncovered with --pass-option-L:
    #     - Add one choice of that comparison + 1 (Representing false-choice for this particular dataset, but might still be true for others.)
    for name, val in t.items():
        values[name].append(1)
        for v in val:
            if v+1 not in values[name]:
                values[name].append(v+1)

    # Branch_dict is a fast way of creating the tree-structure through nested dicts.
    # 'end' is a distinct code-version, depending on threshold comparisons.
    def branch_dict(name, i):
        d = {'name' : name,
              True : [{'name' : 'end', 'id' : i}],
              False : [{'name' : 'end', 'id' : i + 1}]}
        return d

    # Function for walking a tree and growing it.
    # Used to create the branch-tree, from the branch_dicts.
    def walk_tree_and_add_param(start, name, deps, i):
        start_copy = list(start)
        for branch in start_copy:
            deps_copy = dict(deps) #ERROR IN ORIGINAL, DEPS WASN*T INDEPENDENT BETWEEN BRANCHES, this is the fix
            cur_node = branch
            # if the param depends on the current_node, down this branch
            if any(dep == cur_node['name'] for dep, _ in deps_copy.items()):
                bl = deps_copy[cur_node['name']]
                del deps_copy[cur_node['name']]
                walk_tree_and_add_param(cur_node[bl], name, deps_copy, i)
            # else if this is the final dependency add the param to the tree
            # if there are still dependencies to be resolved do nothing
            elif len(deps_copy) == 0:
                if len(start_copy) == 1 and cur_node['name'] == 'end':
                    start[start.index(branch)] = branch_dict(name, i)
                else:
                    start.append(branch_dict(name, i))
                break


    # Run the program once to extract the configurable parameters.
    sizes_cmd = './{} --print-size'.format(program[:-4])
    sizes_res = call_program(sizes_cmd)

    # Extract all dependencies and their boolean value
    branch_info = OrderedDict()
    size_p = re.compile('([^ ]*) \(threshold+ \((.*?)\)\)')
    for line in sizes_res.stdout.readlines():
        m = size_p.search(line)
        if m:
            branch_info[m.group(1)] = dict(((i.strip('!'), not i.startswith('!')) for i in m.group(2).split(' ')))

    # clean away non-used paramters
    for k, _ in branch_info.items():
      branch_info[k] = {n: ds for n, ds in branch_info[k].items() if n in branch_info}

    branch_list = list(branch_info.items())
    i = 0 # we need id to generate unique ids

    branch_tree = []

    # the list below keeps track of processed parameters,
    # so we don't try to add parameters before all their dependencies
    # has been added
    processed = []
    # as long as there are parameters to add, add them
    while branch_list:
        pname, pdeps = branch_list.pop(0)
        # if no dependcies add parameter to root of tree
        if not pdeps:
            branch_tree.append(branch_dict(pname, i))
            processed.append(pname)
            i += 2

        # if all dependencies have alredy been added to tree,
        # add parameter to tree, else save parameter for
        # later processing
        elif all(n.strip('!') in processed for n in pdeps):
            walk_tree_and_add_param(branch_tree, pname, pdeps, i)
            processed.append(pname)
            i += 2
        else:
            branch_list.append((pname, pdeps))

    return (datasets, thresholds, values, branch_tree, branch_info)

# Command to compute bench command to call using sys.
# (From old AutoTuner)
def futhark_bench_cmd(cfg, json, times, tile):
    def sizeOption(size):
        return '--pass-option --size={}={}'.format(size, cfg[size])
    size_options = ' '.join(map(sizeOption, cfg.keys()))
    cmd = 'futhark bench --skip-compilation --exclude-case=notune {} '.format(program)
    if json != None:
        cmd += '--json={} '.format(json.name)
    if times != None:
        if not (any(np.isinf(val) for val in times.values())):
            cmd += '--timeout={} '.format(compute_timeout(times))
    if tile != None:
        cmd += '--pass-option --default-tile-size={} '.format(str(tile))
    cmd += size_options
    return cmd

# Quick command to calculate the current timeout, based on the longest "best" time so far. ( + 1 second, since Futhark is very weird)
def compute_timeout(best):
    timeout = ((np.amax(best.values()) * 20.0) / 1000000.0) + (overhead) # Multiplied by 10 because that is the number of runs in benchmarks.
    if np.isinf(timeout):
        return -1
    else:
        return int(timeout)


# Function to extract names of thresholds in just one branch.
def extract_names_OLD(tree_list):
    all_names = []

    # Since there can be multiple independent branches, each is dealt with independently.
    for tree in tree_list:
        names = extract_names_helper(all_names, tree)
        for name in names:
            # Does not add duplicate names.
            if name not in all_names:
                all_names.append(name)

    return all_names

# Helper to the above function.
def extract_names_helper(names, tree):
    # If we have reached the end by a "True" threshold, we add it to the list.
    if tree[True][0]['name'] == 'end':
        names.append(tree['name'])

    # If we have reached the end by a "False", then we have no more to check.
    if tree[False][0]['name'] == 'end':
        return names

    # If this threshold leads to potentially multiple more paths,
    # recursively check all paths for names.
    for branch in tree[False]:
        res = extract_names_helper(names[:], branch)
        for name in res:
            # Again, avoid duplicates.
            if name not in names and name != 'end':
                names.append(name)

    return names

# Little trick to allow for nice printing in some cases.
def backspace(n):
    sys.stdout.write((b'\x08' * n).decode()) # use \x08 char to go back
    sys.stdout.write(' ' * n)
    sys.stdout.write((b'\x08' * n).decode())

# Utility function to find the depth of a branch.
# Not always the best to use though, as depth is usually
# better determined by len(extract_names(branch)) instead.
def depth_of_branch(tree, depth):
    if tree['name'] == 'end':
        return depth
    else:
        for branch in tree[True]:
            depth = max(depth, depth_of_branch(branch, depth + 1))

        for branch in tree[False]:
            depth = max(depth, depth_of_branch(branch, depth + 1))

        return depth


def get_depth_first_nodes(root):
    nodes = []
    stack = [root]
    while stack:
        cur_node = stack[0]
        stack = stack[1:]
        nodes.append(cur_node)
        for child in cur_node.get_rev_children():
            stack.insert(0, child)
    return nodes


def extract_names(branch):
    names = []
    stack = [branch]
    while stack:
        cur_node = stack[0]
        stack = stack[1:]

        if cur_node['name'] != 'end':
            names.append(cur_node['name'])
        else:
            continue

        for child in cur_node[False]:
            stack.insert(0, child)
        for child in cur_node[True]:
            stack.insert(0, child)
    return names

def extract_versions(branch):
    next_horizontal = 0
    versions = []
    stack = [(branch, [], [])]
    while stack:
        cur_node, versions_before, version_after = stack[0]
        #print("{} had v_b {} and v_a {}".format(cur_node['name'], versions_before, version_after))

        stack = stack[1:]

        if cur_node['name'] == 'end':
            v = versions_before + version_after
            if sum(v) == 0 and any(sum(x) == 0 for x in versions):
                versions = filter(lambda x: sum(x) != 0, versions)
                versions.append(v)
                continue

            versions.append(v)
            #print("ID {} delivered {} . . . {}".format(cur_node['id'], versions_before, version_after))
            continue

        true_names  = sum([len(extract_names(true_branch )) for true_branch  in cur_node[True]])
        false_names = sum([len(extract_names(false_branch)) for false_branch in cur_node[False]])


        for i, child in enumerate(cur_node[False][::-1]):
            v_b = versions_before[:] + [False] + [False for x in range(true_names)]

            for child2 in cur_node[False][::-1][:i]:
                v_b += [False for x in extract_names(child2)]

            v_a = version_after[:]

            for child2 in cur_node[False][::-1][i+1:]:
                v_a = [False for x in extract_names(child2)] + v_a

            stack.insert(0, (child, v_b, v_a))

        for i, child in enumerate(cur_node[True][::-1]):
            v_b = versions_before[:] + [True]

            for child2 in cur_node[True][::-1][:i]:
                v_b += [False for x in extract_names(child2)]

            for child2 in cur_node[True][::-1][i+1:]:
                v_a =  [False for x in extract_names(child2)] + v_a

            v_a = [False for x in range(false_names)] + version_after[:]

            stack.insert(0, (child, v_b, v_a))

    return versions



# Function to extract all "versions" for a branch.
# One version is a list of booleans, each corresponding to a threshold.
# The resulting list-of-lists has one list pr. code-version possible in that branch.
# This means running all of those "versions" results in exhaustive search.
def extract_versions_OLD(depth, tree):
    versions = []

    # If True leads to an end, the remaining thresholds are set to False.
    # This is because each "version" has to have a value for _every_ threshold.
    if tree[True][0]['name'] == 'end':
        versions.append([True] + [False for x in range(depth - 1)])


    # If there are multiple "false" paths, combinations of thresholds are neccesary.
    # This is because each of these paths are independent from each other, much like this one here.
    if len(tree[False]) > 1:
        splits = []
        # Extract versions for each branch independently.
        for i, branch in enumerate(tree[False]):
            splits.append([])
            res = extract_versions_OLD(depth_of_branch(branch, 0), branch)
            for v in res:
                splits[i].append(v)

        # Combine all lists, with the base already computed.
        # Basic code for creating "combinations".
        tmp_versions = [[False]]
        for split in splits:
            new_tmp = []
            for i, version1 in enumerate(split):
                for version2 in tmp_versions:
                    new_tmp.append(version2 + version1)

            tmp_versions = new_tmp
        return tmp_versions
    else:
        # Just 1 False-path exists.
        # If it is an end, we know we have reached the end.
        if tree[False][0]['name'] == 'end':
            versions.append([False])
        else:
            # If it was not the end, we recursively continue.
            res = extract_versions_OLD(depth - 1, tree[False][0])
            for v in res:
                    versions.append([False] + v)

    return versions

# Code to create a string from a list of booleans.
# Useful for indexing "versions" into dicts.
def version_to_string(version):
    res = ''
    for char in [str(int(x)) for x in version]:
        res += char
    return res

# A function for making a string representing an execution path for each dataset.
# Returns a string like the following example:
# D0V3V7D1V2V5D2D2V1V5
# This is from a branch-tree with 2 branches.
# Dataset 0 (D0) chose V3 for the first branch, and V7 for the second.
# Dataset 1 (D1) chose V2 for the first branch, and V5 for the second.
# Dataset 2 (D2) chose V1 for the first branch, and V5 for the second.
def compute_execution_path(threshold_conf):
    # I have an easy look-up list for threshold values in this run.
    # What "end" do I have in each branch of the branch tree.
    result_string = ''

    #print("CONFIG: {}".format(threshold_conf))

    global branch_tree

    for i, dataset in enumerate(datasets):
        result_string += 'D' + str(i)
        for branch in branch_tree:
            part_branch = compute_execution_path_branch(branch, threshold_conf, 'V', dataset)
            result_string += part_branch

    return result_string

# Letter is used to differentiate the kind of version.
# 'V' = Standard branch, default.
# 'B' = Branching branch, aka one with multiple optimizations at the same layer.
# 'L' = Looping branch, aka one with a large amount of intermediate versions.
# 'LE' = Looping Branch End.
def compute_execution_path_branch(branch, threshold_conf, letter, dataset):
    node_name = branch['name']

    threshold_value = threshold_conf[node_name]

    global thresholds
    if len(thresholds[dataset][node_name]) == 1:
        threshold_comparison = threshold_value <= thresholds[dataset][node_name][0]

        chosen_branch = branch[threshold_comparison]
        if len(chosen_branch) == 1:
            # Simple case where we continue on to one other branch.
            chosen_branch = chosen_branch[0]
            if chosen_branch['name'] == 'end':
                ender = letter + str(chosen_branch['id'])
                return ender
            else:
                return compute_execution_path_branch(chosen_branch, threshold_conf, letter, dataset)
        else:
            # We have a nested branching situation.
            # We change the letter to B, and call this function on each branch.
            result_string = ''

            for new_branch in chosen_branch:
                result_string += compute_execution_path_branch(new_branch, threshold_conf, 'B', dataset)

            return (result_string)

    else:
        # WE HAVE A LOOP
        all_true = True
        max_i = -1

        for i, val in enumerate(thresholds[dataset][node_name]):
            #print("i: {}, {} < {}, max_i: {}, all_true: {}".format(i, threshold_value, val, max_i, all_true))
            if not (threshold_value <= val):
                max_i = i
                all_true = False

        if all_true:
            chosen_branch = branch[True]
            if len(chosen_branch) == 1:
                if chosen_branch[0]['name'] == 'end':
                    return ('LE' + str(chosen_branch[0]['id']))
                else:
                    print("A TRUE BRANCH LEAD TO A NON-END")
            else:
                print("A TRUE BRANCH WAS NOT A SINGLETON!")
        else:
            res_string = 'L' + str(max_i)

            chosen_branch = branch[False]
            if len(chosen_branch) == 1:
                if chosen_branch[0]['name'] == 'end':
                    res_string +=('LE' + str(chosen_branch[0]['id']))
                else:
                    res_string += compute_execution_path_branch(chosen_branch[0], threshold_conf, 'L', dataset)
            else:
                # BRANCH IN A LOOP!?
                for new_branch in chosen_branch:
                    res_string += 'LB' + compute_execution_path_branch(new_branch, threshold_conf, 'L', dataset)

            return res_string

# Check if Threshold name 1 is dependent on Threshold name 2 being False.
def dependency_check(T1, T2, dependency_list):
    return (T2 in dependency_list[T1])


#============#
# THE SCRIPT #
#============#=========================================================#
#                                                                      #
# This script is split into three stages.                              #
# The first does the preparation work in the earlier-defined function. #
# The second does the first full benchmarking pass.                    #
# The third breaks all remaining conflicts in the result from stage 2. #
#                                                                      #
#======================================================================#

#========================#
# STAGE 1 - Preparations #
#========================#

script_results = []
if len(sys.argv) == 1:
    programs = ['LocVolCalib.fut', 'bfast-ours.fut', 'variant.fut']
else:
    programs = sys.argv[1:]

for program in programs:
    print_str = "# STARTING TO RUN: {} #".format(program)
    print("#" + '=' * (len(print_str) - 2) + "#")
    print(print_str)
    print("#" + '=' * (len(print_str) - 2) + "#")

    # Log the start-time, used in logging.
    start = time.time()
    num_executed = 0

    # Compile the target program.
    compile_cmd = 'futhark opencl {}'.format(program)
    print('Compiling {}... '.format(program), end='')
    sys.stdout.flush()
    compile_res = call_program(compile_cmd)
    print('Done.')

    # Run the above function to find:
    # Names of all datasets and thresholds.
    # Values of all threshold comparisons.
    # Branch-tree information for dependencies between thresholds.
    (datasets, thresholds, values, branch_tree, dependency_info) = extract_thresholds_and_values(program)

    print("Finished extraction.")

    # Find number of branchesext in the list
    numBranches = len(branch_tree)

    # Extract all the thresholds names, for easier lookup into the dicts.
    #threshold_names = extract_names(branch_tree)
    threshold_names = []
    for branch in branch_tree:
        threshold_names = threshold_names + extract_names(branch)

    # Number of thresholds to optimize
    numThresholds = len(values.keys())

    # Highest comparison-value found.
    max_comparison = 0
    for comparisons in values.values():
        max_comparison = max(max_comparison, np.amax(comparisons))
    max_comparison = max_comparison

    # Prepare the final configuration (and baseline for conflict-resolution) and conflict-dict.
    # These will contain the final result
    base_conf = {}
    conflicts = {}

    # Start by initializing each threshold-range.
    threshold_ranges = {}
    for dataset in datasets:
        threshold_ranges[dataset] = {}
        for name in threshold_names:
            threshold_ranges[dataset][name] = {}
            threshold_ranges[dataset][name]['min'] = 1
            threshold_ranges[dataset][name]['max'] = max_comparison + 1

    # Initialize the merged ranges.
    merged_ranges = {}
    for name in threshold_names:
        merged_ranges[name] = {}
        merged_ranges[name]['min'] = 0
        merged_ranges[name]['max'] = max_comparison + 1

    for name in threshold_names:
        for dataset in datasets:
            # Extract the comparisons made against this threshold
            threshold_list = thresholds[dataset][name]

            if len(threshold_list) > 1:
                print("Name {} have multiple options for dataset {}: {}".format(name, dataset, threshold_list))
                # Consider this a "Conflict" between a lot of possible values.
                # This is not as easy to tune, and will be handled on it's own.
                # Add them to the conflict list, and add the "max" value as the chosen "default" for testing later.
                processed = sorted(threshold_list)[::-1]

                # Avoid overwriting earlier conflicts
                if name not in conflicts:
                    conflicts[name] = []

                for element in processed:
                    # Avoid adding duplicates to the conflict list
                    if (name, element) not in conflicts[name]:
                        conflicts[name].append((name, element))

                # Use maximum as "default"
                max_val = processed[0]

                if name in base_conf:
                    base_conf[name] = max(max_val + 1, base_conf[name])
                else:
                    base_conf[name] = max_val + 1


    for name in threshold_names:
        if name in base_conf:
            base_conf[name] = int(base_conf[name] * 1.5)
            continue

        base_conf[name] = 1

        for dataset in datasets:
            if len(thresholds[dataset][name]) != 0:
                base_conf[name] = max(base_conf[name], thresholds[dataset][name][0])

        base_conf[name] = int(base_conf[name] * 1.5)

    print("Base-configuration found")

    # Potential debug-printing, not used usually.
    #print("")
    #print("Datasets: ")
    #print(datasets)
    #print("")

    print("Thresholds: ")
    print(thresholds)
    print("")

    print("Values: ")
    print(values)
    print("")

    print("Branches: ")
    print(branch_tree)
    print("")

    #print("Max Comparison: {}".format(max_comparison))
    #print("")

    print("Threshold Names: {}".format(threshold_names))
    print("")

    print("Threshold Names DEPTH: {}".format(extract_names(branch_tree[0])))
    print("")

    #print("EXTRACT_VERSIONS")
    #print(extract_versions(branch_tree[0]))



    #=============================================#
    # PREPARE THE FIRST BENCHMARK, AND GET READY! #
    #=============================================#
    execution_cache = {}
    best_times = {}
    best_versions = {}
    baseline_times = {}
    #Perform the FIRST bench-run, to get a base-line and prepare for the proper loop.
    with tempfile.NamedTemporaryFile() as json_tmp:
        # Benchmark using all-false thresholds.
        conf = dict(base_conf)
        for name in threshold_names:
            if name not in conf:
                print("{} WAS NOT IN INITIAL 'BASE_CONF'".format(name))
                conf[name] = max_comparison + 1

        execution_cache[compute_execution_path(conf)] = {}

        print("Starting first Benchmark")
        bench_cmd = futhark_bench_cmd(conf, json_tmp, None, None)

        num_executed += 1
        wall_start = time.time()
        call_program(bench_cmd)
        wall_duration = time.time() - wall_start

        json_data = json.load(json_tmp)

        base_datasets = json_data[program]['datasets']

        dataset_runtimes = []
        for dataset in base_datasets:
            try:
                dataset_runtimes.append(sum(base_datasets[dataset]['runtimes']) / 1000000.0)
                runtime = int(np.mean(base_datasets[dataset]['runtimes']))
                best_times[dataset] = runtime
                best_versions[dataset] = conf
                execution_cache[compute_execution_path(conf)][dataset] = runtime

                baseline_times[dataset] = runtime

            except:
                dataset_runtimes.append(np.inf)
                best_times[dataset] = np.inf
                best_versions[dataset] = conf
                execution_cache[compute_execution_path(conf)][dataset] = np.inf
                baseline_times[dataset] = 1000000

        overhead = (wall_duration - (sum(dataset_runtimes) / len(dataset_runtimes))) + 1
        print("Overhead: {}".format(overhead))



    # Sorted list by branch-depth.
    # Chosen based on the heuristic that deeper branches allow for more impactful tuning.
    order = [(len(extract_names(branch_tree[i])),i) for i, branch in enumerate(branch_tree)]
    deepest_first_order = sorted(order, key=lambda x: x[0])[::-1]

    # Used for pretty-printing.
    num_versions = 0
    for d, i in order:
        num_versions += len(extract_versions(branch_tree[i]))
    current_version_num = 1



    #====================================#
    # EXHAUSTIVE TUNING OF EASY BRANCHES #
    #====================================#

    for depth, i in deepest_first_order:
        # Calculate the number of thresholds before and after this branch.
        depth_before = 0
        for j in range(i):
            depth_before += len(extract_names(branch_tree[j]))

        # Extract all code-versions from this branch.
        branch_names = extract_names(branch_tree[i])
        branch_versions = extract_versions(branch_tree[i])

        # Skip this branch if it contains conflicts already.
        if any(name in conflicts for name in branch_names):
            continue

        # Initialise "best" version as all-false with an impossible time.
        best_branch_version = {}
        best_branch_time = {}
        for dataset in datasets:
            best_branch_time[dataset] = np.inf

        # Loop over every code-version in this branch.
        for j, current_version in enumerate(branch_versions[::-1]):
            status_string = "[{}s] For branch {} trying version {} / {}: {}".format(int(time.time() - start), i, current_version_num, num_versions, current_version)
            print(status_string)
            current_version_num += 1

            # Extract the threshold_names of all the thresholds being tested.
            true_names = [threshold_names[k] if b else ' ' for (k, b) in zip(range(depth_before, depth_before + depth), current_version)]

            # Construct a dict based on the baseline, with the changes that we are trying.
            conf = dict(base_conf)
            bool_to_int = [1 if b else base_conf[name] + 1 for (b, name) in zip(current_version, branch_names)]

            for (name, val) in zip(branch_names, bool_to_int):
                conf[name] = val

            # If an equivalent version has already been run, skip it.
            path = compute_execution_path(conf)
            if path in execution_cache:
                cached_dict = execution_cache[path]

                total_time = 0

                for dataset, runtime in cached_dict.items():
                    total_time += runtime / float(baseline_times[dataset] * 100)

                    if runtime < best_times[dataset]:
                        print("Considered new best for dataset {} at {}".format(dataset, runtime))
                        best_times[dataset] = runtime
                        best_versions[dataset] = conf

                    if runtime < best_branch_time[dataset]:
                        best_branch_time[dataset] = runtime
                        best_branch_version[dataset] = current_version
                continue

            #print("Trying Version: {}".format(conf))

            # Initialize this version in the dict of runtimes.
            execution_cache[path] = {}

            # With a temporary JSON file, run the benchmarking for this version.
            with tempfile.NamedTemporaryFile() as json_tmp:
                bench_cmd = futhark_bench_cmd(conf, json_tmp, best_times, None)

                num_executed += 1
                call_program(bench_cmd)

                json_data = json.load(json_tmp)

                results = json_data[program]['datasets']

                # Time taken is going to be relative to each dataset.
                # This is done by comparing every runtime to the baseline version for this branch.
                # Percentage-improvement is added to the total_time, meaning all datasets are equally important.
                # (This might slightly favour small-runtime datasets, as deviations mean more)
                total_time = 0

                # Record every dataset's runtime, and store it.
                for dataset in results:
                    try:
                        runtime = int(np.mean(results[dataset]['runtimes']))
                        #print("I didn't time out!")
                        execution_cache[path][dataset] = runtime

                        #print("Dataset {} ran in {}, compared to base {}".format(dataset, runtime, baseline_times[dataset]))

                        total_time +=  runtime / float(baseline_times[dataset] * 100)

                        if runtime < best_times[dataset]:
                            print("Considered new best for dataset {} at {}".format(dataset, runtime))
                            best_times[dataset] = runtime
                            best_versions[dataset] = conf

                        if runtime < best_branch_time[dataset]:
                            best_branch_time[dataset] = runtime
                            best_branch_version[dataset] = current_version
                    except:
                        #It timed out on this dataset, or produced wrong results
                        print("Timed out / failed on dataset {}".format(dataset))
                        total_time += np.inf

        print("Finished branch-run, trying to merge ranges.")
        # Modify the base version to use the new "better" version.
        # Also update the baseline-times to be the one for this new baseline configuration
        for dataset, version in best_branch_version.items():
            print("[{}s] Dataset {} prefered version {} with time {}".format(int(time.time() - start), dataset, version, best_branch_time[dataset]))
            # Go through each threshold in the version.
            for i, thresh in enumerate(version):
                # Extract the name and the value it is compared to in this dataset's run.
                # Only a single parameter comparison: Easy!
                # We know it only has a single parameter, because if it didn't it would have been caught as a conflict.
                name = branch_names[i]
                if len(thresholds[dataset][name]) == 0:
                    continue
                val = thresholds[dataset][name][0]

                skipFlag = False
                for TName, T in zip(branch_names[:i], version[:i]):
                    if not T:
                        continue

                    if dependency_check(name, TName, dependency_info):
                        # This means no matter what choice I make, the version has already been chosen so to say.
                        threshold_ranges[dataset][name]['min'] = 1
                        threshold_ranges[dataset][name]['max'] = base_conf[name]
                        skipFlag = True
                        break

                if skipFlag:
                    skipFlag = False
                    continue

                # Use that value as the new min/max depending on whether this threshold was true or not.
                if(thresh):
                    # This was set to True in the configuration
                    # This means, t < Param had to be correct.
                    threshold_ranges[dataset][name]['min'] = 1
                    threshold_ranges[dataset][name]['max'] = val
                else:
                    # This comparison was False in the configuration
                    # This means, t > Param had to be correct.
                    threshold_ranges[dataset][name]['min'] = val + 1
                    threshold_ranges[dataset][name]['max'] = base_conf[name] + 1

        # These are simply the maximum and minimum of all threshold-values, across all datasets.
        for ranges in threshold_ranges.values():
            for name in branch_names:
                merged_ranges[name]['min'] = max(merged_ranges[name]['min'], ranges[name]['min'])
                merged_ranges[name]['max'] = min(merged_ranges[name]['max'], ranges[name]['max'])

        # Check each threshold, to see if it was a conflict or not.
        for name in branch_names:
            if name in conflicts:
                continue

            if(merged_ranges[name]['max'] < merged_ranges[name]['min']):
                # A conflict has occured!
                # Add the two possible values to the conflict dict.
                print("Conflict encountered in {}: {} vs {}".format(name, merged_ranges[name]['min'], merged_ranges[name]['max']))
                conflicts[name] = []
                conflicts[name].append((name, merged_ranges[name]['max']))
                base_conf[name] = merged_ranges[name]['max'] #Just choose one, since both are "good" probably.
                conflicts[name].append((name, merged_ranges[name]['min']))
            else:
                base_conf[name] = int(merged_ranges[name]['max'])


        for name in branch_names[::-1]:
            # If name was added to conflict in the above check.
            # We break this tie now, in order to choose the best of either of the two conflicts.
            # This might not be ideal, but no exhaustive search technique will solve this problem "nicely".
            if name in conflicts and name in merged_ranges:
                # Tie breaking strategy:
                # Simply run both options, and take the best of the two.
                best_option_time = np.inf
                best_option = conflicts[name][0][1]
                for (name, option) in conflicts[name]:
                    conf = dict(base_conf) #Copy baseline values
                    conf[name] = option

                    path = compute_execution_path(conf)
                    if path in execution_cache:
                        total_time = 0
                        for dataset in datasets:
                            if dataset in execution_cache[path]:
                                runtime = execution_cache[path][dataset]
                            else:
                                runtime = np.inf

                            total_time += runtime

                            if runtime < best_times[dataset]:
                                print("Considered new best for dataset {} at {}".format(dataset, runtime))
                                best_times[dataset] = runtime
                                best_versions[dataset] = conf

                            if runtime < best_branch_time[dataset]:
                                best_branch_version[dataset] = current_version

                        print("Option {} for {} ran in {} total time (CACHED)".format(option, name, total_time))

                        if total_time < best_option_time:
                            best_option_time = total_time
                            best_option = option
                            continue
                    else:
                        execution_cache[path] = {}

                    with tempfile.NamedTemporaryFile() as json_tmp:
                        bench_cmd = futhark_bench_cmd(conf, json_tmp, best_times, None)

                        num_executed += 1
                        call_program(bench_cmd)

                        json_data = json.load(json_tmp)

                        results = json_data[program]['datasets']

                        total_time = 0

                        for dataset in results:
                            try:
                                runtime = int(np.mean(results[dataset]['runtimes']))
                                execution_cache[path][dataset] = runtime

                                #print("Dataset {} ran in {}, compared to base {}".format(dataset, runtime, baseline_times[dataset]))

                                total_time +=  runtime

                                if runtime < best_times[dataset]:
                                    print("Considered new best for dataset {} at {}".format(dataset, runtime))
                                    best_times[dataset] = runtime
                                    best_versions[dataset] = conf

                                if runtime < best_branch_time[dataset]:
                                    best_branch_version[dataset] = current_version
                            except:
                                #It timed out on this dataset
                                print("Timed out on dataset {}".format(dataset))
                                total_time += np.inf

                        print("Option {} for {} ran in {} total time".format(option, name, total_time))

                        if total_time < best_option_time:
                            best_option_time = total_time
                            best_option = option

                # Conflict resolved, best_option is saved for use in next branches.
                del(conflicts[name])
                print("Chose the option {} for {}".format(best_option, name))
                base_conf[name] = best_option

        for dataset, runtime in best_times.items():
            print("Dataset {} has seen best {} so far.".format(dataset, runtime))


    #======================#
    # ACTIVE LEARNING      #
    # STRATEGY EXPLANATION #
    #======================#====================================================================#
    # The goal of Active Learning is to make informed decisions about which datapoints to query #
    # Currently, this is relaxed to trying a few uninformed uniform guesses first.              #
    # Afterwards, a "predictive model" of how the objective function looks is estimated roughly #
    # A new point is queried, and the model is validated on how accurately it predicted.        #
    # Repeat a few times until pretty certain of result, and pick best point so far.            #
    #                                                                                           #
    # This is done only for the "difficult" problem of multiple-option thresholds.              #
    #===========================================================================================#

    # Only perform this step if there are variant-size conflicts remaining to fix.
    final_conf = dict(base_conf)
    if len(conflicts) != 0:
        print("Conflicts encountered in:")
        for name, options in conflicts.items():
            pplist = [val for (n, val) in options]
            print("{} with these options: {}".format(name, pplist))


        #print("Encountered the following variant-size conflicts:")
        #print(conflicts)

        for depth, i in deepest_first_order:
            branch = branch_tree[i]
            branch_names = extract_names(branch)

            if not (any(name in conflicts for name in branch_names)):
                continue

            # Extract possible "options" for each threshold involved in this branch.
            options = {}
            numOptions = 0

            # Add all conflict-values to the option list
            for name, val in conflicts.items():
                if name in branch_names:
                    numOptions += len(val)
                    sortedVals = sorted(val, key=lambda x: x[1])
                    sortedVals.append((name, sortedVals[-1][1] + 1))
                    options[name] = [x for (n, x) in sortedVals][::-1]

            # Add the remaining "simple" thresholds (if any)
            for name in branch_names:
                # Skip all the "difficult" ones already covered
                if name in options:
                    continue

                options[name] = ([(name, 1), (name, final_conf[name])]) #Adds the "Always True" and "ALways False" paths


            # Use those options to extract "versions"
            print("Options: {}".format(options))

            branch_versions = extract_versions(branch_tree[i])[::-1]

            # Initialize best-parameters.
            best_branch_time = 0

            print("[{}s] Breaking conflicts in Branch {}, found {} conflicts over {} thresholds.".format(int(time.time() - start), i, numOptions, len(options)))

            # Run each "version" to find the best one.
            for j, version in enumerate(branch_versions):
                # First, find the deepest True threshold, indicating the current tuning parameter
                position = -1
                for i, bool in enumerate(version):
                    if bool:
                        position = i

                if position == -1:
                    # All was false!
                    # We have nothing to tune here, so we don't.
                    continue

                print("[{}s] Finding best value for threshold {}".format(int(time.time() - start), branch_names[position]))

                # Find the index of this threshold in Options
                name = branch_names[position]

                conf = dict(final_conf) #Copy the "fixed" values

                best_threshold_value = 0
                best_threshold_time  = np.inf

                loop_options = list(options[name])

                first = 0
                last = len(loop_options) - 1
                first_iteration = True

                if False: # len(loop_options) > 5:
                    #print("Starting Active Tuning")
                    # ACTIVE LEARNING USEFUL!!!
                    # We try to do active learning....
                    polynomial_features= PolynomialFeatures(degree=2)
                    model = LinearRegression()
                    
                    numGuesses = 3
                    guesses_tried = []
                    runtimes = {}

                    x_all = np.array(loop_options)[:, np.newaxis]
                    x_all_poly = polynomial_features.fit_transform(x_all)

                    #guesses = np.random.choice(len(loop_options), numGuesses, replace=False).astype(int)
                    guesses = list(np.random.choice(range(len(loop_options) - 2), 3, replace=False))
                    #guesses = guesses + [0, int(len(loop_options) / 2.0), len(loop_options) - 1]
                    guesses_tried += list(guesses)
                    if not first_iteration:
                        guesses_tried = guesses_tried + [len(loop_options) - 1]
                    else:
                        first_iteration = False
                    print("Starting with: {}".format(guesses_tried))


                    x = [loop_options[i] for i in guesses]
                    x_poly = polynomial_features.fit_transform(np.array(x)[:,np.newaxis])

                    y = []
                    for g in guesses:
                        conf[name] = loop_options[g]
                        with tempfile.NamedTemporaryFile() as json_tmp:
                            bench_cmd = futhark_bench_cmd(conf, json_tmp, None, 16)

                            num_executed += 1
                            call_program(bench_cmd)
                            json_data = json.load(json_tmp)
                            results = json_data[program]['datasets']

                            total_time = 0
                            for dataset in results:
                                runtime = int(np.mean(results[dataset]['runtimes']))
                                total_time +=  runtime

                            y.append(total_time)
                            runtimes[loop_options[g]] = total_time
                        print("Attempting {} giving f({}) = {}".format(g, g, total_time))

                    x_poly = polynomial_features.fit_transform(np.array(x)[:,np.newaxis])
                    model = LinearRegression()
                    model.fit(x_poly, np.array(y)[:, np.newaxis])
                    predictions = [int(e) for e in list(model.predict(x_all_poly).T[0])]
                    candidate = np.argmin(predictions)
                    old_candidate = -2
                    while old_candidate != candidate:
                        print(predictions)
                        print("Predicted: {} with f({}) = {}".format(candidate, candidate, predictions[candidate]))
                        old_candidate = candidate

                        # Predict a new low-point, the lowest "predicted" so far.
                        # This candidate was given from prior iteration of loop.
                        guesses_tried.append(candidate)

                        # Get the "true" benchmark for that guess
                        x.append(loop_options[candidate])

                        conf[name] = loop_options[candidate]
                        with tempfile.NamedTemporaryFile() as json_tmp:
                            bench_cmd = futhark_bench_cmd(conf, json_tmp, None, 16)

                            num_executed += 1
                            call_program(bench_cmd)
                            json_data = json.load(json_tmp)
                            results = json_data[program]['datasets']

                            total_time = 0
                            for dataset in results:
                                runtime = int(np.mean(results[dataset]['runtimes']))
                                total_time +=  runtime

                            y.append(total_time)
                            runtimes[loop_options[candidate]] = total_time

                        print("Actual : {}".format(total_time))
                        # Train a final model with this new point as well, and pick the final point
                        x_poly = polynomial_features.fit_transform(np.array(x)[:,np.newaxis])
                        model = LinearRegression()
                        model.fit(x_poly, np.array(y)[:, np.newaxis])

                        # Predict the next candidate.
                        # If this is the earlier one, then we stop.
                        predictions = [int(e) for e in list(model.predict(x_all_poly).T[0])]
                        candidate = np.argmin(predictions)

                    winner_index = candidate
                    winner_T = loop_options[winner_index]

                    if winner_T in runtimes:
                        winner_R = runtimes[winner_T]
                    else:
                        with tempfile.NamedTemporaryFile() as json_tmp:
                            conf[name] = winner_T
                            bench_cmd = futhark_bench_cmd(conf, json_tmp, None, 16)

                            num_executed += 1
                            guesses_tried.append(winner_T)
                            call_program(bench_cmd)
                            json_data = json.load(json_tmp)
                            results = json_data[program]['datasets']

                            total_time = 0
                            for dataset in results:
                                runtime = int(np.mean(results[dataset]['runtimes']))
                                total_time +=  runtime

                            winner_R = total_time

                    print("[{}s] Chose {} giving {} with runtimes {} using {} runs for threshold {}".format(int(time.time() - start), winner_index, winner_T, winner_R, len(guesses_tried), name))
                    conf[name] = winner_T
                    best_threshold_value = winner_T

                elif len(loop_options) > 5:
                    # Binary-Search Choice
                    numBin = 0
                    while len(loop_options[first:last]) != 0:
                        numBin += 1
                        if first_iteration:
                            midpoint = 0
                        else:
                            midpoint = (first + last) // 2

                        val = loop_options[midpoint]
                        print("[{}s] Trying value {} for threshold {}".format( int(time.time() - start), val, name ))

                        # Update the specific conflict-threshold's value to this possible option.
                        conf[name] = val

                        # Run the benchmark.
                        with tempfile.NamedTemporaryFile() as json_tmp:
                            bench_cmd = futhark_bench_cmd(conf, json_tmp, None, 16)

                            num_executed += 1
                            call_program(bench_cmd)

                            json_data = json.load(json_tmp)

                            results = json_data[program]['datasets']

                            # This time using total aggregate runtime.
                            # This was chosen since earlier we optimized based on datasets, and here we don't.
                            # (Aggregate runtime favours longer-running datasets)
                            total_time = 0
                            for dataset in results:
                                try:
                                    runtime = int(np.mean(results[dataset]['runtimes']))

                                    total_time +=  runtime

                                    #print("[{}s] Dataset {} ran in {}".format(int(time.time() - start), dataset, runtime))

                                except:
                                    # It timed out on this dataset
                                    # This means I add the total "best" to this one, as it can't be better anyway.
                                    total_time += np.inf

                        # Update "best" overall version.
                        if total_time < best_branch_time:
                            best_branch_time = total_time
                            best_branch = conf

                        # If first iteration of binary search, set stuff.
                        if first_iteration:
                            print("[{}s] Current best {} is {} with {}".format(int(time.time() - start), name, val, total_time))
                            best_threshold_time = total_time
                            best_threshold_value = val
                            first_iteration = False
                            first = 1
                            continue

                        # Update "best" current threshold value
                        if total_time < best_threshold_time:
                            print("[{}s] Current best {} is {} with {}".format(int(time.time() - start), name, val, total_time))
                            best_threshold_time = total_time
                            best_threshold_value = val
                            first = midpoint + 1
                        else:
                            last = midpoint - 1
                    print("BINARY SEARCH USED {} EXECUTIONS FOR T: {}".format(numBin, name))
                else:
                    # Exhaustive-Search Choice
                    # Just try every version, since there are so few.

                    plot_list_x = []
                    plot_list_y = []

                    for k, current_option in enumerate(options[name]):

                        # Due to sloppy coding, current_option can either be a tuple (name, val) or it can be just val....
                        # So we try one, and if the unpacking of the tuple fails, then it's the other. 
                        # This is purely sloppy coding, sorry.
                        try:
                            namehere, val = current_option
                            name = namehere
                        except:
                            val = current_option

                        print("[{}s] Trying value {} for threshold {}".format( int(time.time() - start), val, name ))

                        # Update the specific conflict-threshold's value to this possible option.
                        conf[name] = val

                        # Run the benchmark.
                        with tempfile.NamedTemporaryFile() as json_tmp:
                            if i == 0:
                                bench_cmd = futhark_bench_cmd(conf, json_tmp, None, 16)
                            else:
                                bench_cmd = futhark_bench_cmd(conf, json_tmp, None, 16)

                            num_executed += 1
                            call_program(bench_cmd)

                            json_data = json.load(json_tmp)

                            results = json_data[program]['datasets']

                            # This time using total aggregate runtime.
                            # This was chosen since earlier we optimized based on datasets, and here we don't.
                            # (Aggregate runtime favours longer-running datasets)
                            total_time = 0
                            for dataset in results:
                                try:
                                    runtime = int(np.mean(results[dataset]['runtimes']))

                                    total_time +=  runtime

                                    #print("[{}s] Dataset {} ran in {}".format(int(time.time() - start), dataset, runtime))

                                except:
                                    # It timed out on this dataset
                                    # This means I add the total "best" to this one, as it can't be better anyway.
                                    total_time += np.inf

                        # Update "best" overall version.
                        if total_time < best_branch_time:
                            best_branch_time = total_time
                            best_branch = conf

                        # Update "best" current threshold value
                        if total_time < best_threshold_time:
                            print("Current best {} is {} with {}".format(name, val, total_time))
                            best_threshold_time = total_time
                            best_threshold_value = val

                        plot_list_x.append(val)
                        plot_list_y.append(total_time)

                    # Printing for a plot thingy, not used.
                    #print("")
                    #print(name)
                    #print(plot_list_x)
                    #print(plot_list_y)
                    #print("")

                print("[{}s] Chose the following threshold for {} : {}".format(int(time.time() - start), name, best_threshold_value))
                final_conf[name] = best_threshold_value

    best_tile = 16

    best_tile_time = np.inf

    print("Skipping tile-calib, just debug")

    tiles = [4, 8, 16, 32, 64]

    print("[{}s]Starting Tile-Size Calibration: {}".format(int(time.time() - start), tiles))
    for tile in tiles:
        print("[{}s] Trying Tile: {}".format(int(time.time() - start), tile))
        with tempfile.NamedTemporaryFile() as json_tmp:
            bench_cmd = futhark_bench_cmd(final_conf, json_tmp, None, tile)
            num_executed += 1
            call_program(bench_cmd)

            json_data = json.load(json_tmp)
            results = json_data[program]['datasets']
            total_time = 0

            for dataset in results:
                try:
                    runtime = int(np.mean(results[dataset]['runtimes']))
                    total_time +=  runtime

                    #print("[{}s] Dataset {} ran in {} with tile-size {}".format(int(time.time() - start), dataset, runtime, tile))

                    if best_times[dataset] > runtime:
                        best_times[dataset] = runtime


                except:
                    # It timed out on this dataset
                    # This means I add the total "best" to this one, as it can't be better anyway.
                    total_time += np.inf

            if total_time < best_tile_time:
                print("Chose new best tile-size at {} with {} compared to old {}".format(tile, total_time, best_tile_time))
                best_tile = tile
                best_tile_time = total_time

    # Report the results
    script_results.append( (time.time() - start, futhark_bench_cmd(final_conf, None, None, best_tile), num_executed) )



# PRINT ALL RESULTS!
print("")
print_str = "# FINISHED RUNNING {} BENCHMARKS #".format(len(programs))
print("#" + '=' * (len(print_str) - 2) + "#")
print(print_str)
print("#" + '=' * (len(print_str) - 2) + "#")

for i, program in enumerate(programs):
    (time_taken, bench_cmd, num_executed) = script_results[i]
    print("")
    print("Final command for target program {}, took {}s, with {} executions".format(program[:-4], int(time_taken), num_executed))
    print(bench_cmd.replace(' --exclude-case=notune ', ' '))
    
print("Saving all final benchmarks in JSON files for results.")
for i, program in enumerate(programs):
    print("Saving results of {}".format(program))
    (time_taken, bench_cmd, num_executed) = script_results[i]
    call_program(bench_cmd.replace(' --exclude-case=notune ', ' ') + ' --json=binary-{}.json'.format(program[:-4]))



"""
#===============#
# NOTES SECTION #
#===============#


#===========#
# TILE SIZE #
#===========#

#======#
# SRAD #
#======#

#=============#
# LocVolCalib #
#=============#

#=======#
# BFAST #
#=======#

#=====#
# LUD #
#=====#

#======================#
# VARIANT-SIZE TESTING #
#======================#
 """
