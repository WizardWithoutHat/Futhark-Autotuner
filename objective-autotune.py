#!/usr/bin/env python2
#
# A simple autotuner for calibrating parameters of Futhark programs.
# Based on OpenTuner.  Requires Python 2.

#================================================#
# Original version of the Futhark Auto Tuner     #
# With a slight tweak to generating search-space #
#================================================#

from __future__ import print_function

import opentuner
from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter, LogIntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from opentuner.search import technique, bandittechniques, simplextechniques

import sys
import tempfile
import json
import os
import re
import math
import itertools
import bisect
import time
import sys
import logging
from collections import OrderedDict, defaultdict

# We define our own LogIntegerParameter to crudely work around a bug
# in OpenTuner: https://github.com/jansel/opentuner/issues/114
class FixedLogIntegerParameter(LogIntegerParameter):
  def _scale(self, v):
    v = max(self.min_value, v) # WORKAROUND
    return math.log(v + 1.0 - self.min_value, 2.0)

technique.register(bandittechniques.AUCBanditMetaTechnique([
    simplextechniques.RegularNelderMead(),
    simplextechniques.RightNelderMead(),
    simplextechniques.MultiNelderMead(),
    simplextechniques.RandomTorczon(),
    simplextechniques.RegularTorczon(),
    simplextechniques.RightTorczon(),
    simplextechniques.MultiTorczon()
], name='FutharkMetaTechnique'))

class FutharkTuner(MeasurementInterface):
  def __init__(self, args, *pargs, **kwargs):

    kwargs['program_name'] = args.program

    # if no techniques are specified by the user
    # stick to hill-climbing techniques
    if args.technique == None:
      args.technique = ['FutharkMetaTechnique']

    # if calculating time-out only generate one config at the time
    if args.calc_timeout:
      args.parallelism = 1

    super(FutharkTuner, self).__init__(args, *pargs, **kwargs)

    self.log = logging.getLogger(__name__)
    self.branch_tree = []
    self.thresholds = OrderedDict()
    self.baseline = defaultdict(int)
    self.datasets = []
    self.values = defaultdict(set)
    self.covered_branches = {}
    self.timeout_branches = defaultdict(list)

    # We only compile the program once, since the parameters we are
    # tuning can be given at run-time.  This saves a lot of time for
    # programs that take a long time to compile.
    compile_cmd = '{} {} {}'.format(self.args.futhark, self.args.backend, self.program_name())
    print('Compiling {}... '.format(self.program_name()), end='')
    sys.stdout.flush()
    compile_res = self.call_critical_program(compile_cmd)
    print('Done.')

    if self.args.calc_timeout:
      print('Calculating --timeout based on untuned run... ', end='')
      sys.stdout.flush()
      self.args.timeout = self.get_base_timeout()
      print('{} seconds.'.format(self.args.timeout))

    print('Extracting threshold parameters and values... ', end='')
    sys.stdout.flush()
    n = self.extract_thresholds()

    print('{} threshold comparisons detected.'.format(n))

    if n == 0:
      print('\nProgram does not have any threshold parameters.  Nothing to tune.')
      sys.exit(0)

    print('Calculating values for each threshold parameter... ', end='')
    sys.stdout.flush()
    n = self.yield_values()
    print('{} threshold values generated.'.format(n))

    print('Building the branching tree... ', end='')
    sys.stdout.flush()
    n = self.build_branch_tree()
    print('{} branches generated.'.format(n))


  def call_critical_program(self, cmd):
    r = self.call_program(cmd)
    if r['returncode'] != 0:
      print("Command '%s' failed with stdout:" % (cmd))
      print(r['stdout'])
      print('And stderr:')
      print(r['stderr'])
      sys.exit(1)
    return r

  def get_base_timeout(self):
    """
    Run a benchmark without specifying thresholds,
    in order to use the runtime of the slowest dataset
    as timeout-value, plus the (estimated) overhead of initialisation.
    """
    with tempfile.NamedTemporaryFile() as json_tmp:
      base_cmd = '{} bench {} -e {} --exclude-case=notune --backend=opencl --skip-compilation --json={}'.format(
        self.args.futhark, self.program_name(), self.args.entry_point, json_tmp.name)
      wall_start = time.time()
      base_res = self.call_critical_program(base_cmd)
      wall_duration = time.time() - wall_start

      json_data = json.load(json_tmp)
      base_datasets = json_data[self.program_key_name()]['datasets']
      high = 0
      dataset_runtimes = [ sum(base_datasets[dataset]['runtimes']) / 1000000.0
                           for dataset in base_datasets ]
      high = max(dataset_runtimes)
      overhead = (wall_duration - sum(dataset_runtimes)) / len(dataset_runtimes)

    return int(math.ceil(high + overhead)) + 1 # Extra second for luck.

  def extract_thresholds(self):
    """
    Extract the threshold parameters to be tuned
    along with the different values, that each
    parameter are compared against for each dataset.
    """
    sizes_cmd = './{} --print-sizes'.format(self.program_bin_name())
    sizes_res = self.call_critical_program(sizes_cmd)

    # the tuner needs all params for all datasets,
    # so extract and save all params, so we can add them
    # later, if they are not used in comparison
    all_params = []
    size_p = re.compile('([^ ]*) \(threshold+ \((.*?)\)\)')
    for line in sizes_res['stdout'].splitlines():
      m = size_p.search(line)
      if m:
        all_params.append(m.group(1))

    n = 0
    with tempfile.NamedTemporaryFile() as json_tmp:
      # extract comparison values
      val_cmd = '{} bench {} -e {} --exclude-case=notune --backend=opencl --skip-compilation --pass-option=-L --runs=1 --json={}'.format(
        self.args.futhark, self.program_name(), self.args.entry_point, json_tmp.name)
      val_res = self.call_critical_program(val_cmd)

      json_data = json.load(json_tmp, object_pairs_hook=OrderedDict)

      datasets = json_data[self.program_key_name()]['datasets']

      # search for parameters and values for each dataset
      val_re = re.compile('Compared ([^ ]+) <= (-?\d+)')
      for dataset in datasets:
        self.baseline[dataset] = datasets[dataset]['runtimes'][0]
        self.thresholds[dataset] = defaultdict(set)
        for line in datasets[dataset]['stderr'].splitlines():
          match = val_re.search(line)
          if match:
            param, value = match.group(1), int(match.group(2))
            # Add comparison; note there might be several per parameter.
            self.thresholds[dataset][param].add(value)
            n += 1

        # if a param has not been used in a comparison,
        # and thus not added, add it now
        for p in all_params:
          if not p in self.thresholds[dataset]:
            self.thresholds[dataset][p] = set()

      # we need a list of datasets later
      self.datasets = datasets.keys()
    return n

  # since we only need the min/max of the possible values,
  # this function could be simplified, but as an possible
  # exhaustive technique would need all the values,
  # and since the calculations do not hurt performance,
  # it is kept in this form for now
  def yield_values(self):
      """
      Calculate all configurations based
      on the threshold values such that
      all possible combinations are tried.
      """
      # extract threshold values for each parameter across all datasets
      # and save to a dict with all threshold values for each parameter
      # keeping the threshold values sorted
      t = defaultdict(list)
      for dataset, thresholds in self.thresholds.items():
        for k, vs in thresholds.items():
          for v in vs:
            if v not in t[k]:
              bisect.insort(t[k], v)

      # iterate over all sorted threshold values of all parameters
      # and calculate a value lying between the current and previous value
      n=0
      for name, values in t.items():
        for i in range(0,len(values)):
          current_value = values[i]
          prev_value = values[i - 1] if i != 0 else 0
          v = int((current_value + prev_value) / 2)
          n += 1
          self.values[name].add(v)

        # for the last threshold value also calculate a larger value
        n += 1
        v = int((values[len(values) - 1] * 2))
        self.values[name].add(v)
      return n

  def build_branch_tree(self):
    """
    Extract information about dependencies among the threshold
    parameters, and build a branching-tree, which can be used
    to exclude configurations that ends in the same branch etc.
    """
    def branch_dict(name, i):
      d = {'name' : name,
           True : [{'name' : 'end', 'id' : i}],
           False : [{'name' : 'end', 'id' : i + 1}]}
      return d

    def walk_tree_and_add_param(start, name, deps, i):
      start_copy = list(start)
      for branch in start_copy:
        cur_node = branch
        # if the param depends on the current_node, down this branch
        if any(dep == cur_node['name'] for dep, _ in deps.items()):
          bl = deps[cur_node['name']]
          del deps[cur_node['name']]
          walk_tree_and_add_param(cur_node[bl], name, deps, i)
        # else if this is the final dependency add the param to the tree
        # if there are still dependencies to be resolved do nothing
        elif len(deps) == 0:
          if len(start_copy) == 1 and cur_node['name'] == 'end':
            start[start.index(branch)] = branch_dict(name, i)
          else:
            start.append(branch_dict(name, i))
          break

    # Run the program once to extract the configurable parameters.
    sizes_cmd = './{} --print-size'.format(self.program_bin_name())
    sizes_res = self.call_critical_program(sizes_cmd)

    # extract all dependencies and their boolean value
    branch_info = OrderedDict()
    size_p = re.compile('([^ ]*) \(threshold+ \((.*?)\)\)')
    for line in sizes_res['stdout'].splitlines():
      m = size_p.search(line)
      if m:
        branch_info[m.group(1)] = dict(((i.strip('!'), not i.startswith('!')) for i in m.group(2).split(' ')))

    # clean away non-used paramters
    for k, _ in branch_info.items():
      branch_info[k] = {n: ds for n, ds in branch_info[k].items() if n in branch_info}

    branch_list = list(branch_info.items())
    i = 0 # we need id to generate unique ids

    # the list below keeps track of processed parameters,
    # so we don't try to add parameters before all their dependencies
    # has been added
    processed = []
    # as long as there are parameters to add, add them
    while branch_list:
      pname, pdeps = branch_list.pop(0)
      # if no dependcies add parameter to root of tree
      if not pdeps:
        self.branch_tree.append(branch_dict(pname, i))
        processed.append(pname)
        i += 2

      # if all dependencies have alredy been added to tree,
      # add parameter to tree, else save parameter for
      # later processing
      elif all(n.strip('!') in processed for n in pdeps):
        walk_tree_and_add_param(self.branch_tree, pname, pdeps, i)
        processed.append(pname)
        i += 2
      else:
        branch_list.append((pname, pdeps))
    return i

  def program_bin_name(self):
    return os.path.splitext(self.program_name())[0]

  def program_key_name(self):
    if self.args.entry_point == 'main':
      return self.program_name()
    else:
      return self.program_name() + ':' + self.args.entry_point

  def interesting_class(self, p_class):
    return len(self.args.only) == 0 or p_class in self.args.only

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """
    manipulator = ConfigurationManipulator()
    for k, v in self.values.items():
      manipulator.add_parameter(FixedLogIntegerParameter(k, min(v), max(v)))

    return manipulator

  def futhark_bench_cmd(self, cfg):
    def sizeOption(size):
      return '--pass-option --size={}={}'.format(size, cfg[size])
    size_options = ' '.join(map(sizeOption, cfg.keys()))
    def otherOption(opt):
      return '--pass-option {}'.format(opt)
    other_options = ' '.join(map(otherOption, self.args.pass_option))
    return '{} bench --skip-compilation {} --exclude-case=notune {} {}'.format(
      self.args.futhark, self.program_name(), size_options, other_options)

  def get_branch(self, start, cfg, datasets, result):
    """
    Calculate the branch for a given cfg on datasets
    """
    for d in datasets:
      for branch in start:
        node = branch
        if node['name'] != 'end':
          bl = all(cfg[node['name']] <= t for t in self.thresholds[d][node['name']])
          self.get_branch(node[bl], cfg, [d], result)
        else:
          result.append(node['id'])
    return tuple(result)

  def run(self, desired_result, input, limit):
    """
    Compile and run a given configuration then
    return performance
    """
    branches = []
    # calculate the branch for every branch, to see if it is reported as a time-out
    # if so, return a timeout
    for d in self.datasets:
      b = self.get_branch(self.branch_tree, desired_result.configuration.data, [d], [])
      if b in self.timeout_branches[d]:
        return Result(state='TIMEOUT', time=float('inf'))
      else:
        branches.append(b)
    branches = tuple(branches)

    # check if the combined execution path of all programs has already been tried
    # if so return old result
    # if not actually try configuration
    if branches in self.covered_branches:
      return Result(time=self.covered_branches[branches])

    self.log.info('New distinct configuration: {}'.format(', '.join(map(lambda (k,v): k + '=' + str(v), desired_result.configuration.data.items()))))

    with tempfile.NamedTemporaryFile() as bench_json:
      bench_cmd = '{} --json {} --timeout {}'.format(
        self.futhark_bench_cmd(desired_result.configuration.data),
        bench_json.name, self.args.timeout)

      # we need to catch timeouts in order to blacklist
      # branches, thus we do not assert the result,
      # but instead handle errors in a try-clause
      run_res = self.call_program(bench_cmd)

      # Sum all the runtimes together to quantify the performance of
      # this configuration.  This may be too crude, as it heavily
      # favours the longer-running data sets.
      json_data = json.load(bench_json)
      datasets = json_data[self.program_key_name()]['datasets']
      runtime = 0
      for dataset in datasets:
        try:
          run = datasets[dataset]['runtimes']
          meanRun = sum(run) / len(run)
          base = self.baseline[dataset]
          
          meanRun = (float(meanRun) / base) * 100
        
          runtime += meanRun
      # if a timeout/error has occured add the offending branch to
      # the pool of timeout branches
        except TypeError:
          self.timeout_branches[dataset].append(self.get_branch(self.branch_tree, desired_result.configuration.data, [dataset], []))
          return Result(state='TIMEOUT', time=float('inf'))

    self.covered_branches[branches] = runtime
    return Result(time=runtime)

  def save_final_config(self, configuration):
    """called at the end of tuning"""
    filename = self.args.save_json
    if filename != None:
      print("Optimal parameter values written to %s: %s" % (filename, configuration.data))
      self.manipulator().save_to_file(configuration.data, filename)
    else:
      print("--save-json not given, so not writing parameter values to file.")
    print("Reproduce with command:")
    print(self.futhark_bench_cmd(configuration.data))

if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  argparser.add_argument('program', type=str, metavar='PROGRAM')
  argparser.add_argument('--backend', type=str, metavar='BACKEND', default='opencl')
  argparser.add_argument('--futhark', type=str, metavar='FUTHARK', default='futhark')
  argparser.add_argument('--timeout', type=int, metavar='TIMEOUT', default='60')
  argparser.add_argument('--entry-point', '-e', type=str, metavar='NAME', default='main')
  argparser.add_argument('--pass-option', type=str, metavar='OPTION', action='append', default=[])
  argparser.add_argument('--save-json', type=str, metavar='FILENAME', default=None)
  argparser.add_argument('--calc-timeout', action='store_true')

  args = argparser.parse_args()

FutharkTuner.main(args)
