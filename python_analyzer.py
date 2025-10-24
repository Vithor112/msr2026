import os
import subprocess
import json
import re
import pandas as pd
from collections import Counter
from base_analyzer import BaseAnalyzer

class PythonAnalyzer(BaseAnalyzer):
    """
    A concrete implementation of BaseAnalyzer for Python.
    Uses locstats, flake8, bandit, and pylint.
    """
    
    def __init__(self, repo_path, git_handler):
        """
        Initializes the PythonAnalyzer.

        Args:
            repo_path (str): The local path to the cloned repository.
            git_handler (GitHandler): An instance of GitHandler.
        """
        super().__init__(repo_path, git_handler)
        self._cached_flake8_output = None # Cache for flake8 results per commit

    def get_language(self):
        """Implements abstract method."""
        return 'python'

    def analyze_pr_deltas(self, base_sha, head_sha):
        """
        Orchestrates the full analysis of a PR for Python.
        Checks out base, analyzes, checks out head, analyzes, and diffs.
        """
        print(f"  Analyzing delta between base({base_sha[:7]}) and head({head_sha[:7]})...")
        
        print(f"  Checking out base: {base_sha[:7]}...")
        if not self.git_handler.checkout_and_clean(self.repo_path, base_sha):
            print(f"  Failed to checkout base SHA: {base_sha}. Skipping PR.")
            return None
        
        self._cached_flake8_output = None # Clear cache
        base_metrics = self.build_metric_dict()

        print(f"  Checking out head: {head_sha[:7]}...")
        if not self.git_handler.checkout_and_clean(self.repo_path, head_sha):
            print(f"  Failed to checkout head SHA: {head_sha}. Skipping PR.")
            return None

        self._cached_flake8_output = None # Clear cache
        head_metrics = self.build_metric_dict()

        print("  Calculating deltas...")
        delta_dict = {}

        simple_metric_keys = [
            'loc', 'cyclomatic_sum', 'over_cyclomatic_count',
            'cognitive_sum', 'over_cognitive_count',
            'security_syntax_errors', 'security_severity_high',
            'security_severity_medium', 'security_severity_low',
            'general_fatal', 'general_error', 'general_warning'
        ]
        
        for key in simple_metric_keys:
            base_val = base_metrics.get(key)
            head_val = head_metrics.get(key)
            delta_key = f'delta_{key}'
            
            if base_val is not None and head_val is not None:
                delta_dict[delta_key] = head_val - base_val
            else:
                delta_dict[delta_key] = pd.NA

        base_sec_all = base_metrics.get('security_all_errors_counter', Counter())
        head_sec_all = head_metrics.get('security_all_errors_counter', Counter())
        base_sec_high = base_metrics.get('security_high_sev_errors_counter', Counter())
        head_sec_high = head_metrics.get('security_high_sev_errors_counter', Counter())

        new_sec_all = head_sec_all - base_sec_all
        removed_sec_all = base_sec_all - head_sec_all
        new_sec_high = head_sec_high - base_sec_high
        removed_sec_high = base_sec_high - head_sec_high

        self._populate_top_n_deltas(delta_dict, 'new_security', new_sec_all, 3)
        self._populate_top_n_deltas(delta_dict, 'removed_security', removed_sec_all, 3)
        self._populate_top_n_deltas(delta_dict, 'new_security_high_sev', new_sec_high, 1)
        self._populate_top_n_deltas(delta_dict, 'removed_security_high_sev', removed_sec_high, 1)

        # c) General (Pylint) Counter deltas
        base_gen_all = base_metrics.get('general_errors_counter', Counter())
        head_gen_all = head_metrics.get('general_errors_counter', Counter())

        new_gen_all = head_gen_all - base_gen_all
        removed_gen_all = base_gen_all - head_gen_all

        self._populate_top_n_deltas(delta_dict, 'new_general', new_gen_all, 3)
        self._populate_top_n_deltas(delta_dict, 'removed_general', removed_gen_all, 3)

        return delta_dict

    # --- Implementation of Abstract Analysis Methods ---

    def analyze_loc(self):
        """
        Calculates Lines of Code (LOC) using locstats.
        Calls _set_loc.
        """
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            result = subprocess.run(['locstats', self.language, '.', '-m'], check=True, capture_output=True, text=True)
            
            loc_output = result.stdout.strip()
            if not loc_output:
                print("  Error: 'locstats' produced no output.")
                self._set_loc(None)
                return
                
            last_line = loc_output.split('\n')[-1]
            self._set_loc(int(last_line.strip()))
            
        except FileNotFoundError:
            print("  'locstats' command not found. Please install it ('pip install locstats').")
            self._set_loc(None)
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"  Error running 'locstats' or parsing output: {e}")
            self._set_loc(None)
        finally:
            os.chdir(original_dir)

    def _run_flake8_once(self):
        """
        A private helper to run flake8 once and cache the output lines.
        """
        if self._cached_flake8_output is not None:
            return self._cached_flake8_output

        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            command = "flake8 --max-complexity 0 --max-cognitive-complexity=0 . | grep complex || true"
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            
            output_lines = result.stdout.strip().split('\n')
            
            if not output_lines or (len(output_lines) == 1 and not output_lines[0]):
                self._cached_flake8_output = [] # No complexity issues
            else:
                self._cached_flake8_output = output_lines
                
            return self._cached_flake8_output

        except FileNotFoundError:
            print("  'flake8' command not found. Please install it ('pip install flake8 flake8-cognitive-complexity').")
            self._cached_flake8_output = [] # Treat as no issues
            return []
        except subprocess.CalledProcessError as e:
            print(f"  Error running 'flake8': {e.stderr.strip()}")
            self._cached_flake8_output = [] # Treat as no issues
            return []
        finally:
            os.chdir(original_dir)

    def analyze_complexity(self):
        """
        Analyzes cyclomatic and cognitive complexity using flake8.
        Relies on _run_flake8_once() to get the data.
        Calls _set_cyclomatic_complexity and _set_cognitive_complexity.
        """
        output_lines = self._run_flake8_once()
        
        # Cyclomatic
        total_cyclomatic_complexity = 0
        over_cyclomatic_count = 0
        cyclomatic_regex = re.compile(r"\((\d+)\)$") # For C901, e.g., (8)

        # Cognitive
        total_cognitive_complexity = 0
        over_cognitive_count = 0
        cognitive_regex = re.compile(r"\((\d+) > \d+\)$") # For CCR001, e.g., (11 > 0)

        if output_lines:
            for line in output_lines:
                # Cyclomatic
                if 'C901' in line.strip():
                    try:
                        match = cyclomatic_regex.search(line.strip())
                        if match:
                            complexity = int(match.group(1))
                            total_cyclomatic_complexity += complexity
                            if complexity > 10: # Cyclomatic threshold
                                over_cyclomatic_count += 1
                    except (IndexError, ValueError) as e:
                        print(f"  Warning: Error parsing C901 line: '{line.strip()}'. Error: {e}")
                
                # Cognitive
                if 'CCR001' in line.strip():
                    try:
                        match = cognitive_regex.search(line.strip())
                        if match:
                            complexity = int(match.group(1))
                            total_cognitive_complexity += complexity
                            if complexity > 7: # Cognitive threshold
                                over_cognitive_count += 1
                    except (IndexError, ValueError) as e:
                        print(f"  Warning: Error parsing CCR001 line: '{line.strip()}'. Error: {e}")
        
        self._set_cyclomatic_complexity(total_cyclomatic_complexity, over_cyclomatic_count)
        self._set_cognitive_complexity(total_cognitive_complexity, over_cognitive_count)

    def analyze_security_faults(self):
        """
        Analyzes for security issues using bandit.
        Calls _set_security_metrics and _set_security_counters.
        """
        syntax_errors = 0
        high, medium, low = 0, 0, 0
        all_errors_counter = Counter()
        high_sev_errors_counter = Counter()
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            result = subprocess.run(['bandit', '-r', '-f', 'json', '.'], capture_output=True, text=True)
            bandit_output = result.stdout

            if not bandit_output:
                if result.stderr: print(f"  Error running 'bandit': {result.stderr.strip()}")
                else: print("  Error running 'bandit': No output produced.")
                # Still call setters with default values
                self._set_security_metrics(syntax_errors, high, medium, low)
                self._set_security_counters(all_errors_counter, high_sev_errors_counter)
                return

            try:
                json_start_index = bandit_output.find('{')
                if json_start_index == -1:
                    raise json.JSONDecodeError("No JSON object start found.", bandit_output, 0)
                
                json_string = bandit_output[json_start_index:]
                data = json.loads(json_string)
                
            except json.JSONDecodeError as e:
                print(f"  Error parsing bandit JSON output: {e}")
                # Still call setters with default values
                self._set_security_metrics(syntax_errors, high, medium, low)
                self._set_security_counters(all_errors_counter, high_sev_errors_counter)
                return

            syntax_errors = len(data.get('errors', []))

            metrics_data = data.get('metrics', {})
            for file_metrics in metrics_data.values():
                high += file_metrics.get('SEVERITY.HIGH', 0)
                medium += file_metrics.get('SEVERITY.MEDIUM', 0)
                low += file_metrics.get('SEVERITY.LOW', 0)

            results_data = data.get('results', [])
            for issue in results_data:
                test_name = issue.get('test_name', 'UNKNOWN')
                severity = issue.get('issue_severity', 'UNKNOWN')
                key = f"{test_name}_{severity}"
                all_errors_counter[key] += 1
                
                if severity == 'HIGH':
                    high_sev_errors_counter[key] += 1
            
            self._set_security_metrics(syntax_errors, high, medium, low)
            self._set_security_counters(all_errors_counter, high_sev_errors_counter)

        except FileNotFoundError:
            print("  'bandit' command not found. Please install it ('pip install bandit').")
            self._set_security_metrics(syntax_errors, high, medium, low)
            self._set_security_counters(all_errors_counter, high_sev_errors_counter)
        except Exception as e:
            print(f"  An unexpected error occurred during bandit analysis: {e}")
            self._set_security_metrics(syntax_errors, high, medium, low)
            self._set_security_counters(all_errors_counter, high_sev_errors_counter)
        finally:
            os.chdir(original_dir)

    def analyze_general_faults(self):
        """
        Analyzes for errors using pylint.
        Calls _set_general_fault_metrics and _set_general_fault_counters.
        """
        fatal, error, warning = 0, 0, 0
        pylint_errors_counter = Counter()
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            command = [
                'pylint',
                '--errors-only',
                # Disabling common false positives in complex repos
                '--disable=import-error,undefined-variable,no-member,no-name-in-module',
                '--output-format', 'json',
                '.'
            ]
            # We use json format, not json2, as json is just the list of messages
            # which is easier to parse reliably.
            result = subprocess.run(command, capture_output=True, text=True)
            pylint_output = result.stdout

            if not pylint_output:
                if result.stderr: print(f"  Error running 'pylint': {result.stderr.strip()}")
                else: print("  'pylint' produced no output (likely no issues).")
                # Call setters with defaults
                self._set_general_fault_metrics(fatal, error, warning)
                self._set_general_fault_counters(pylint_errors_counter)
                return

            try:
                json_start_index = pylint_output.find('[')
                if json_start_index == -1:
                    if pylint_output.strip() == "": 
                        self._set_general_fault_metrics(fatal, error, warning)
                        self._set_general_fault_counters(pylint_errors_counter)
                        return # No issues
                    raise json.JSONDecodeError("No JSON array start found.", pylint_output, 0)

                json_string = pylint_output[json_start_index:]
                messages = json.loads(json_string)

            except json.JSONDecodeError as e:
                print(f"  Error parsing pylint JSON output: {e}")
                self._set_general_fault_metrics(fatal, error, warning)
                self._set_general_fault_counters(pylint_errors_counter)
                return

            for msg in messages:
                msg_type = msg.get('type', 'unknown')
                symbol = msg.get('symbol', 'U')

                if msg_type == 'fatal':
                    fatal += 1
                elif msg_type == 'error':
                    error += 1
                elif msg_type == 'warning':
                    warning += 1
                
                key = f"{msg_type}_{symbol}"
                pylint_errors_counter[key] += 1
            
            self._set_general_fault_metrics(fatal, error, warning)
            self._set_general_fault_counters(pylint_errors_counter)

        except FileNotFoundError:
            print("  'pylint' command not found. Please install it ('pip install pylint').")
            self._set_general_fault_metrics(fatal, error, warning)
            self._set_general_fault_counters(pylint_errors_counter)
        except Exception as e:
            print(f"  An unexpected error occurred during pylint analysis: {e}")
            self._set_general_fault_metrics(fatal, error, warning)
            self._set_general_fault_counters(pylint_errors_counter)
        finally:
            os.chdir(original_dir)
