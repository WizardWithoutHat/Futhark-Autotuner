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
import cma

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

    return (datasets, thresholds, values, branch_tree)

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
        cmd += '--timeout={} '.format(compute_timeout(times))

    if tile != None:
        cmd += '--pass-option --default-tile-size={} '.format(str(tile))

    cmd += size_options

    return cmd

# Quick command to calculate the current timeout, based on the longest "best" time so far. ( + 1 second, since Futhark is very weird)
def compute_timeout(best):
    global overhead
    return int((np.amax(best.values()) * 20) / 1000000.0) + int(overhead) # Multiplied by 10 because that is the number of runs in benchmarks.


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


def nearest_power_of_2(x):
    if x <= 0:
        return 1
    return 2 ** int(np.log2(x))


# Evaluation Function, to be run with CMA-ES
def evaluation_function(threshold_list, baseconf, name_list):
    #Convert input list to conf
    conf = baseconf
    
    global eparPrThreshold

    for name, val in zip(name_list, threshold_list):
        epars = eparPrThreshold[name]
        if val > ( len(epars) - 1 ):
            valC = epars[-1]
        elif val < 0:
            valC = epars[0]
        else:
            valC = epars[int(round(val))]
            
        conf[name] = valC

    tile = 16

    # Find all execution paths for this configuration.
    # Allows look-up of earlier computations.
    execution_path = compute_execution_path(conf)# + 'T' + str(tile)

    global execution_cache
    if execution_path in execution_cache:
        global num_skipped
        num_skipped += 1

        return execution_cache[execution_path]
    else:
        global num_executed
        num_executed += 1

    global start
    print("[{}s] Attempting execution path: {} with TILE: {}".format(int(time.time() - start), execution_path, tile))

    with tempfile.NamedTemporaryFile() as json_tmp:
        global baseline_times
        if num_executed == 1:
            bench_cmd = futhark_bench_cmd(conf, json_tmp, None, tile)
        else:
            bench_cmd = futhark_bench_cmd(conf, json_tmp, baseline_times, tile)

        #print(bench_cmd)
        call_program(bench_cmd)

        json_data = json.load(json_tmp)

        results = json_data[program]['datasets']

        total_time = 0

        # Record every dataset's runtime, and store it.
        global datasets
        try:
            for dataset in results:
                runtime = int(np.mean(results[dataset]['runtimes']))

                if num_executed == 1:
                    baseline_times[dataset] = runtime

                #print("[{}s] Dataset {} ran in {}".format(int(time.time() - start), dataset, runtime))

                total_time += float(runtime) / float(baseline_times[dataset])
        except:
            print("FAILED!")
            global num_failed
            num_failed += 1
            total_time += 99999

        execution_cache[execution_path] = total_time
        execution_cache[execution_path + 'VERS'] = bench_cmd

    return total_time

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


def futhark_autotune_program(program):
    #========================#
    # STAGE 1 - Preparations #
    #========================#
    # Log the start-time, used in logging.
    global start
    start = time.time()

    print_str = "# STARTING TO RUN: {} #".format(program)
    print("#" + '=' * (len(print_str) - 2) + "#")
    print(print_str)
    print("#" + '=' * (len(print_str) - 2) + "#")

    # Compile the target program.
    #program = sys.argv[1]
    compile_cmd = 'futhark opencl {}'.format(program)
    print('Compiling {}... '.format(program), end='')
    sys.stdout.flush()
    compile_res = call_program(compile_cmd)
    print('Done.')

    # Run the above function to find:
    # Names of all datasets and thresholds.
    # Values of all threshold comparisons.
    # Branch-tree information for dependencies between thresholds.
    global datasets
    global branch_tree
    global thresholds
    (datasets, thresholds, values, branch_tree) = extract_thresholds_and_values(program)

    # Sorted list by branch-depth.
    # Chosen based on the heuristic that deeper branches allow for more impactful tuning.
    order = [(len(extract_names(branch_tree[i])),i) for i, branch in enumerate(branch_tree)]
    deepest_first_order = sorted(order, key=lambda x: x[0])[::-1]

    # Extract all the thresholds names, for easier lookup into the dicts.
    global threshold_names
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

    # Potential debug-printing, not used usually.
    print("")
    print("Datasets: ")
    print(datasets)
    print("")

    print("Thresholds: ")
    print(thresholds)
    print("")

    print("Values: ")
    print(values)
    print("")

    print("Branches: ")
    print(branch_tree)
    print("")

    print("Max Comparison: {}".format(max_comparison))
    print("")

    print("Threshold Names: {}".format(threshold_names))
    print("")

    global eparPrThreshold
    eparPrThreshold = {}
    for dataset in datasets:
        for T, vals in thresholds[dataset].items():
            if T not in eparPrThreshold:
                eparPrThreshold[T] = set()
            
            for v in vals:
                eparPrThreshold[T].add(v)

    for T, epars in eparPrThreshold.items():
        eparPrThreshold[T] = list(epars)
        eparPrThreshold[T].sort()
        eparPrThreshold[T] = eparPrThreshold[T] + [eparPrThreshold[T][-1] + 1]


    #================================#
    # STAGE 2 - Run CMA-ES algorithm #
    #================================#
    global execution_cache
    execution_cache = {}

    global baseline_times
    baseline_times = {}

    #Perform the FIRST bench-run, to get a base-line and prepare for the proper loop.
    with tempfile.NamedTemporaryFile() as json_tmp:
        # Benchmark using all-false thresholds.
        print("Starting first Benchmark")

        #bench_cmd = futhark_bench_cmd(conf, json_tmp, None, None)
        bench_cmd = "futhark bench {} --skip-compilation --json={}".format(program, json_tmp.name)

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

                baseline_times[dataset] = runtime * 2

            except:
                print("NO-TUNED BENCHMARK FAILED!")
                return("NO-TUNED BENCHMARK FAILED!")
#                dataset_runtimes.append(np.inf)
 #               execution_cache[compute_execution_path(conf)][dataset] = np.inf
  #              baseline_times[dataset] = 1000000 #Not inf, since this is can be anything really...

        #dataset_runtimes = [ sum(base_datasets[dataset]['runtimes']) / 1000000.0
        #                        for dataset in base_datasets ]

        global overhead
        overhead = (wall_duration - (sum(dataset_runtimes) / len(dataset_runtimes))) + 1
        print("Overhead: {}".format(overhead))

    baseline = []
    max_length = 0
    for name in threshold_names:
        baseline.append(np.random.randint(len(eparPrThreshold[name]) - 1))
        #baseline.append((len(eparPrThreshold[name])) - 1) 
        max_length = max(max_length, len(eparPrThreshold[name]))
        
    for T, vals in eparPrThreshold.items():
        print(T)
        print(vals)
        print("")

    global num_skipped
    global num_executed
    global num_failed
    num_skipped = 0
    num_executed = 0
    num_failed = 0

    timeout_val = 60 * 60 * 2

    for depth, i in deepest_first_order:
        # Calculate the number of thresholds before and after this branch.
        depth_before = 0
        for j in range(i):
            depth_before += len(extract_names([branch_tree[j]]))

        baseconf = {}
        for j, (name, val) in enumerate(zip(threshold_names, baseline)):
            if j < depth_before or j > depth_before + depth - 1:
                baseconf[name] = val

        branch_threshold_names = threshold_names[depth_before:depth_before+depth]

        branch_result = cma.fmin(evaluation_function,
                                 baseline[depth_before:depth_before+depth],
                                 int(max_length * 0.4),
#                                 {'popsize': 4 + int(3 * np.log2(len(baseline[depth_before:depth_before+depth]))), 'timeout': timeout_val},
                                 {'timeout': timeout_val},
                                 args=([baseconf, branch_threshold_names]))

        baseline[depth_before:depth_before+depth] = branch_result[0]

    print("[{}s] Skipped {} total executions by caching".format(int(time.time() - start), num_skipped))
    print("[{}s] Performed {} total experiments".format(int(time.time() - start), num_executed))
    if num_failed > 0:
        print("[{}s] Had {} failed executions".format(int(time.time() - start), num_failed))

    # Report the results.
    print("FINAL BENCH COMMAND:")

    conf = {}
    for name, val in zip(threshold_names, baseline):
        epars = eparPrThreshold[name]
        if val > ( len(epars) - 1 ):
            valC = epars[-1]
        elif val < 0:
            valC = epars[0]
        else:
            valC = epars[int(round(val))]
            
        conf[name] = valC


    final_command = futhark_bench_cmd(conf, None, None, 16)#nearest_power_of_2(x[0][-1]))
    print(final_command)

    return final_command

#=========================#
# Actual script being run #
#=========================#===================================================#
# It simply instructs which programs to try to benchmark.                     #
# Currently it either runs the standard tests if no specific input is given.  #
# If specific input is given, it runs all programs given as command-line args #
#=========================#===================================================#

results = []
if len(sys.argv) == 1:
    programs = ['LocVolCalib.fut', 'bfast-ours.fut', 'variant.fut']
else:
    programs = sys.argv[1:]

for program in programs:
    results.append(futhark_autotune_program(program))

for i, program in enumerate(programs):
    print("")
    print("Final command for target program " + program[:-4])
    print(results[i])

print("Saving all final benchmarks in JSON files for results.")
for i, program in enumerate(programs):
    print("Saving results of {}".format(program))
    bench_cmd = results[i]
    call_program(bench_cmd.replace(' --exclude-case=notune ', ' --exclude-case=tune ') + ' --json=cmaindex-{}.json'.format(program[:-4]))



"""
#=================#
# CMA OPTION LIST #
#=================#

 cma.CMAOptions('')
 {'CMA_cmean': '1  # learning rate for the mean value',
  'CMA_dampsvec_fac': 'np.Inf  # tentative and subject to changes, 0.5 would be a "default" damping for sigma vector update',
  'CMA_sampler_options': '{}  # options passed to `CMA_sampler` class init as keyword arguments',
  'verbose': '3  #v verbosity e.g. of initial/final message, -1 is very quiet, -9 maximally quiet, may not be fully implemented',
  'is_feasible': 'is_feasible  #v a function that computes feasibility, by default lambda x, f: f not in (None, np.NaN)',
  'CSA_clip_length_value': 'None  #v poorly tested, [0, 0] means const length N**0.5, [-1, 1] allows a variation of +- N/(N+2), etc.',
  'tolfacupx': '1e3  #v termination when step-size increases by tolfacupx (diverges). That is, the initial step-size was chosen far too small and better solutions were found far away from the initial solution x0',
  'CMA_dampsvec_fade': '0.1  # tentative fading out parameter for sigma vector update',
  'BoundaryHandler': 'BoundTransform  # or BoundPenalty, unused when ``bounds in (None, [None, None])``',
  'fixed_variables': 'None  # dictionary with index-value pairs like {0:1.1, 2:0.1} that are not optimized',
  'CMA_rankone': '1.0  # multiplier for rank-one update learning rate of covariance matrix',
  'CMA_sampler': 'None  # a class or instance that implements the interface of `cma.interfaces.StatisticalModelSamplerWithZeroMeanBaseClass`',
  'typical_x': 'None  # used with scaling_of_variables',
  'verb_append': '0  # initial evaluation counter, if append, do not overwrite output files',
  'vv': '{}  #? versatile set or dictionary for hacking purposes, value found in self.opts["vv"]',
  'minstd': '0  #v minimal std (scalar or vector) in any coordinate direction, cave interference with tol*',
  'CSA_dampfac': '1  #v positive multiplier for step-size damping, 0.3 is close to optimal on the sphere',
  'CSA_disregard_length': 'False  #v True is untested, also changes respective parameters',
  'verb_disp': '100  #v verbosity: display console output every verb_disp iteration',
  'CMA_const_trace': 'False  # normalize trace, 1, True, "arithm", "geom", "aeig", "geig" are valid',
  'AdaptSigma': 'True  # or False or any CMAAdaptSigmaBase class e.g. CMAAdaptSigmaTPA, CMAAdaptSigmaCSA',
  'randn': 'np.random.randn  #v randn(lam, N) must return an np.array of shape (lam, N), see also cma.utilities.math.randhss',
  'verb_filenameprefix': 'outcmaes  # output filenames prefix',
  'integer_variables': '[]  # index list, invokes basic integer handling: prevent std dev to become too small in the given variables',
  'mean_shift_line_samples': 'False #v sample two new solutions colinear to previous mean shift',
  'scaling_of_variables': 'None  # depreciated, rather use fitness_transformations.ScaleCoordinates instead (or possibly CMA_stds).\n            Scale for each variable in that effective_sigma0 = sigma0*scaling. Internally the variables are divided by scaling_of_variables and sigma is unchanged, default is `np.ones(N)`',
  'CMA_active': 'True  # negative update, conducted after the original update',
  'CMA_diagonal': '0*100*N/popsize**0.5  # nb of iterations with diagonal covariance matrix, True for always',
  'tolfun': '1e-11  #v termination criterion: tolerance in function value, quite useful',
  'CMA_mirrors': 'popsize < 6  # values <0.5 are interpreted as fraction, values >1 as numbers (rounded), otherwise about 0.16 is used',
  'maxfevals': 'inf  #v maximum number of function evaluations',
  'maxiter': '100 + 150 * (N+3)**2 // popsize**0.5  #v maximum number of iterations',
  'tolconditioncov': '1e14  #v stop if the condition of the covariance matrix is above `tolconditioncov`',
  'verb_log': '1  #v verbosity: write data to files every verb_log iteration, writing can be time critical on fast to evaluate functions',
  'CSA_damp_mueff_exponent': '0.5  # zero would mean no dependency of damping on mueff, useful with CSA_disregard_length option',
  'transformation': 'None  # depreciated, use cma.fitness_transformations.FitnessTransformation instead.\n            [t0, t1] are two mappings, t0 transforms solutions from CMA-representation to f-representation (tf_pheno),\n            t1 is the (optional) back transformation, see class GenoPheno',
  'verb_plot': '0  #v in fmin(): plot() is called every verb_plot iteration',
  'CMA_recombination_weights': 'None  # a list, see class RecombinationWeights, overwrites CMA_mu and popsize options',
  'tolstagnation': 'int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations',
  'pc_line_samples': 'False #v one line sample along the evolution path pc',
  'timeout': 'inf  #v stop if timeout seconds are exceeded, the string "2.5 * 60**2" evaluates to 2 hours and 30 minutes',
  'signals_filename': 'None  # cma_signals.in  # read versatile options from this file which contains a single options dict, e.g. ``{"timeout": 0}`` to stop, string-values are evaluated, e.g. "np.inf" is valid',
  'updatecovwait': 'None  #v number of iterations without distribution update, name is subject to future changes',
  'tolx': '1e-11  #v termination criterion: tolerance in x-changes',
  'termination_callback': 'None  #v a function returning True for termination, called in `stop` with `self` as argument, could be abused for side effects',
  'CMA_stds': 'None  # multipliers for sigma0 in each coordinate, not represented in C, makes scaling_of_variables obsolete',
  'CMA_eigenmethod': 'np.linalg.eigh  # or cma.utils.eig or pygsl.eigen.eigenvectors',
  'tolfunhist': '1e-12  #v termination criterion: tolerance in function value history',
  'tolupsigma': '1e20  #v sigma/sigma0 > tolupsigma * max(eivenvals(C)**0.5) indicates "creeping behavior" with usually minor improvements',
  'maxstd': 'inf  #v maximal std in any coordinate direction',
  'mindx': '0  #v minimal std in any arbitrary direction, cave interference with tol*',
  'popsize': '4+int(3*np.log(N))  # population size, AKA lambda, number of new solution per iteration',
  'CMA_mirrormethod': '2  # 0=unconditional, 1=selective, 2=selective with delay',
  'CMA_elitist': 'False  #v or "initial" or True, elitism likely impairs global search performance',
  'CMA_teststds': 'None  # factors for non-isotropic initial distr. of C, mainly for test purpose, see CMA_stds for production',
  'seed': 'time  # random number seed for `numpy.random`; `None` and `0` equate to `time`, `np.nan` means "do nothing", see also option "randn"',
  'bounds': '[None, None]  # lower (=bounds[0]) and upper domain boundaries, each a scalar or a list/vector',
  'ftarget': '-inf  #v target function value, minimization',
  'verb_time': 'True  #v output timings on console',
  'CSA_squared': 'False  #v use squared length for sigma-adaptation ',
  'CMA_mu': 'None  # parents selection parameter, default is popsize // 2',
  'CMA_on': '1  # multiplier for all covariance matrix updates',
  'CMA_rankmu': '1.0  # multiplier for rank-mu update learning rate of covariance matrix'}

 """
