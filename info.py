import os
import pandas as pd
import requests
import subprocess
from datasets import load_dataset
import json
from pprint import pprint
import re 
from collections import Counter

github_token = os.environ.get('GITHUB_TOKEN')

def get_pr_base_and_head(repo_full_name, pr_number):
    """
    Fetches the base and head commit sha for a given pull request using the GitHub API.

    Args:
        repo_full_name (str): The full name of the repository (e.g., 'owner/repo').
        pr_number (int): The number of the pull request.

    Returns:
        tuple: A tuple containing the base commit sha and the head commit sha.
               Returns (None, None) if the API call fails.
    """
    headers = {}
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    api_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    
    print(f"Fetching PR details from: {api_url}")
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        pr_data = response.json()
        base_sha = pr_data['base']['sha']
        head_sha = pr_data['head']['sha']
        return base_sha, head_sha
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PR data from GitHub API: {e}")
        if not github_token:
            print("This could be due to rate limiting. You can set a GITHUB_TOKEN environment variable.")
        return None, None
    except KeyError:
        print("Error: 'base' or 'head' sha not found in API response.")
        return None, None

def clone_repo(repo_full_name):
    """
    Checks if a repository is accessible, then clones it if it doesn't already exist.
    """
    clone_url = f"https://github.com/{repo_full_name}.git"
    repo_name = repo_full_name.split('/')[-1]

    # --- Pre-check: Verify repository is accessible ---
    print(f"Verifying accessibility of '{repo_full_name}'...")
    try:
        # Use ls-remote to check repo existence without cloning.
        # --exit-code returns 0 on success, non-zero on failure.
        # -h limits to just heads (e.g., refs/heads/main) for a faster check.
        
        # Add GIT_TERMINAL_PROMPT=0 to the environment to prevent git from hanging on auth prompts
        env = os.environ.copy()
        env['GIT_TERMINAL_PROMPT'] = '0'
        
        subprocess.run(
            ['git', 'ls-remote', '--exit-code', '-h', clone_url],
            check=True, capture_output=True, text=True, env=env
        )
        print("Repository is accessible.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to access repository '{repo_full_name}'. It may be private, deleted, or renamed.")
        print(f"Stderr: {e.stderr.strip()}")
        return None
    # --- End Pre-check ---

    if not os.path.exists(repo_name):
        print(f"Cloning '{repo_full_name}' into './{repo_name}'...")
        try:
            # Also add GIT_TERMINAL_PROMPT=0 to the clone command
            env = os.environ.copy()
            env['GIT_TERMINAL_PROMPT'] = '0'
            subprocess.run(['git', 'clone', clone_url], check=True, capture_output=True, text=True, env=env)
            print("Repository cloned successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e.stderr}")
            return None
    else:
        print(f"Repository '{repo_name}' already exists. Skipping clone.")
    
    return repo_name

class Analyzer:
    """
    A class to analyze a git repository at specific commits for code metrics.
    It is designed to be easily expandable with new analysis methods.
    """
    def __init__(self, repo_path, language):
        """
        Initializes the Analyzer.
        Args:
            repo_path (str): The local path to the cloned repository.
            language (str): The primary programming language of the repository.
        """
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        self.repo_path = repo_path
        self.language = language.lower()

    def _checkout_and_clean(self, commit_sha):
        """Checks out a specific commit and cleans the working directory."""
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            # Use capture_output to suppress stdout/stderr unless there's an error
            subprocess.run(['git', 'checkout', commit_sha, '--force'], check=True, capture_output=True, text=True)
            subprocess.run(['git', 'clean', '-fd'], check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Error checking out commit {commit_sha[:7]}: {e.stderr.strip()}")
            return False
        finally:
            os.chdir(original_dir)

    def _analyze_loc(self):
        """
        Calculates Lines of Code (LOC) for the current state of the repo using locstats.
        Returns:
            int: The total lines of code, or None on failure.
        """
        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            # -m flag produces machine-readable output.
            result = subprocess.run(['locstats', self.language, '.', '-m'], check=True, capture_output=True, text=True)
            
            loc_output = result.stdout.strip()
            if not loc_output:
                print("  Error: 'locstats' produced no output.")
                return None
                
            # Get the last line of the output, as previous lines might be warnings
            last_line = loc_output.split('\n')[-1]
            return int(last_line.strip())
            
        except FileNotFoundError:
            print("  'locstats' command not found. Please install it ('pip install locstats') and ensure it's in your PATH.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"  Error running 'locstats': {e.stderr.strip()}")
            return None
        except ValueError as e:
            print(f"  Error converting locstats output to integer: {e}")
            loc_output_full = result.stdout.strip()
            print(f"  locstats output was: '{loc_output_full}'")
            if '\n' in loc_output_full:
                # Fix: Assign the part with the backslash to a variable first
                last_line_from_output = loc_output_full.split('\n')[-1]
                print(f"  Attempted to parse last line: '{last_line_from_output}'")
            return None
        finally:
            os.chdir(original_dir)

    def _analyze_complexity(self):
        """
        Calculates aggregate cyclomatic and cognitive complexity and counts 
        over-complex methods for the current state of the repo using flake8.
        
        This method is designed for Python repositories.
        
        Returns:
            tuple: (total_cyclomatic, over_cyclomatic_count, 
                    total_cognitive, over_cognitive_count), or (None, None, None, None) on failure.
        """
        # This analysis is specific to Python
        if self.language != 'python':
            return 0, 0, 0, 0 # Return 0 for non-python repos

        original_dir = os.getcwd()
        total_cyclomatic_complexity = 0
        over_cyclomatic_count = 0
        total_cognitive_complexity = 0
        over_cognitive_count = 0
        
        try:
            os.chdir(self.repo_path)
            # Run with both complexity checks. Pipe to grep 'complex' as requested.
            # Add '|| true' so the command doesn't fail if grep finds no matches.
            command = "flake8 --max-complexity 0 --max-cognitive-complexity=0 . | grep complex || true"
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            
            output_lines = result.stdout.strip().split('\n')
            
            if not output_lines or (len(output_lines) == 1 and not output_lines[0]):
                # No complexity warnings found, which means complexity is 0
                return 0, 0, 0, 0

            # Regexes to find the complexity numbers
            cyclomatic_regex = re.compile(r"\((\d+)\)$") # For C901, e.g., (8)
            cognitive_regex = re.compile(r"\((\d+) > \d+\)$") # For CCR001, e.g., (11 > 0)

            for line in output_lines:
                if not line:
                    continue
                
                line_stripped = line.strip()

                if 'C901' in line_stripped: # Cyclomatic Complexity
                    try:
                        match = cyclomatic_regex.search(line_stripped)
                        if match:
                            complexity = int(match.group(1))
                            total_cyclomatic_complexity += complexity
                            if complexity > 10: # Cyclomatic threshold
                                over_cyclomatic_count += 1
                        else:
                             print(f"  Warning: Could not parse C901 from flake8: '{line_stripped}'")
                    except (IndexError, ValueError) as e:
                        print(f"  Warning: Error parsing C901 line: '{line_stripped}'. Error: {e}")
                
                elif 'CCR001' in line_stripped: # Cognitive Complexity
                    try:
                        match = cognitive_regex.search(line_stripped)
                        if match:
                            complexity = int(match.group(1))
                            total_cognitive_complexity += complexity
                            if complexity > 7: # Cognitive threshold
                                over_cognitive_count += 1
                        else:
                             print(f"  Warning: Could not parse CCR001 from flake8: '{line_stripped}'")
                    except (IndexError, ValueError) as e:
                        print(f"  Warning: Error parsing CCR001 line: '{line_stripped}'. Error: {e}")

            return total_cyclomatic_complexity, over_cyclomatic_count, total_cognitive_complexity, over_cognitive_count

        except FileNotFoundError:
            print("  'flake8' command not found. Please install it ('pip install flake8 flake8-cognitive-complexity') and ensure it's in your PATH.")
            return None, None, None, None
        except subprocess.CalledProcessError as e:
            print(f"  Error running 'flake8': {e.stderr.strip()}")
            return None, None, None, None
        finally:
            os.chdir(original_dir)

    def _analyze_security(self):
        """
        Analyzes the repo for security issues using bandit.
        This method is designed for Python repositories.
        
        Returns:
            tuple: (security_metrics_dict, all_errors_counter, high_sev_errors_counter)
        """
        
        # Prepare default metrics
        security_metrics = {
            'python_bandit_syntax_errors': 0,
            'python_bandit_severity_high': 0,
            'python_bandit_severity_medium': 0,
            'python_bandit_severity_low': 0,
        }
        
        empty_counters = (Counter(), Counter())

        # This analysis is specific to Python
        if self.language != 'python':
            return security_metrics, *empty_counters

        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            # Run bandit. Don't use check=True, as bandit exits non-zero on findings.
            result = subprocess.run(['bandit', '-r', '-f', 'json', '.'], capture_output=True, text=True)

            bandit_output = result.stdout

            # Handle cases where bandit fails unexpectedly or produces no output.
            if not bandit_output:
                if result.stderr:
                     print(f"  Error running 'bandit': {result.stderr.strip()}")
                else:
                     print("  Error running 'bandit': No output produced.")
                return security_metrics, *empty_counters # Return defaults

            try:
                # Find the start of the JSON object to skip any leading lines/warnings
                json_start_index = bandit_output.find('{')
                if json_start_index == -1:
                    # If no '{' found, it's not valid JSON.
                    raise json.JSONDecodeError("No JSON object start found in bandit output.", bandit_output, 0)
                
                # Slice the string from the first '{' to the end
                json_string = bandit_output[json_start_index:]
                data = json.loads(json_string)
                
            except json.JSONDecodeError as e:
                print(f"  Error parsing bandit JSON output: {e}")
                print(f"  Bandit output (raw) was: {bandit_output[:500]}...") # Log snippet
                return security_metrics, *empty_counters # Return defaults

            # 1. Syntax Errors (syntax_error_counter)
            security_metrics['python_bandit_syntax_errors'] = len(data.get('errors', []))

            # 2. Aggregated Severity from 'metrics'
            sev_high = 0
            sev_med = 0
            sev_low = 0
            metrics_data = data.get('metrics', {})
            for file_metrics in metrics_data.values():
                sev_high += file_metrics.get('SEVERITY.HIGH', 0)
                sev_med += file_metrics.get('SEVERITY.MEDIUM', 0)
                sev_low += file_metrics.get('SEVERITY.LOW', 0)
            
            security_metrics['python_bandit_severity_high'] = sev_high
            security_metrics['python_bandit_severity_medium'] = sev_med
            security_metrics['python_bandit_severity_low'] = sev_low

            # 3. Top errors from 'results'
            results_data = data.get('results', [])
            all_errors_counter = Counter()
            high_sev_errors_counter = Counter()

            for issue in results_data:
                test_name = issue.get('test_name', 'UNKNOWN')
                severity = issue.get('issue_severity', 'UNKNOWN')
                
                # Key for top 3: test_name + issue_severity
                key = f"{test_name}_{severity}"
                all_errors_counter[key] += 1
                
                # Key for top 1 high severity: test_name
                if severity == 'HIGH':
                    high_sev_errors_counter[test_name] += 1

            # 4. Return metrics dict and the raw counters
            return security_metrics, all_errors_counter, high_sev_errors_counter

        except FileNotFoundError:
            print("  'bandit' command not found. Please install it ('pip install bandit') and ensure it's in your PATH.")
            return security_metrics, *empty_counters # Return defaults
        except Exception as e:
            # Catch other potential errors during analysis
            print(f"  An unexpected error occurred during bandit analysis: {e}")
            return security_metrics, *empty_counters # Return defaults
        finally:
            os.chdir(original_dir)

    def _analyze_pylint(self):
        """
        Analyzes the repo for errors using pylint.
        This method is designed for Python repositories.
        
        Returns:
            tuple: (pylint_metrics_dict, pylint_errors_counter)
        """
        pylint_metrics = {
            'python_pylint_fatal': 0,
            'python_pylint_error': 0,
            'python_pylint_warning': 0,
        }
        pylint_errors_counter = Counter()
        
        # This analysis is specific to Python
        if self.language != 'python':
            return pylint_metrics, pylint_errors_counter

        original_dir = os.getcwd()
        try:
            os.chdir(self.repo_path)
            # Run pylint. Don't check exit code, as it's non-zero for issues.
            command = [
                'pylint',
                '--errors-only',
                '--disable=import-error,undefined-variable,no-member',
                '--output-format', 'json2',
                '.'
            ]
            result = subprocess.run(command, capture_output=True, text=True)

            pylint_output = result.stdout

            if not pylint_output:
                if result.stderr:
                    print(f"  Error running 'pylint': {result.stderr.strip()}")
                else:
                    print("  Error running 'pylint': No output produced.")
                return pylint_metrics, pylint_errors_counter

            try:
                # Find the start of the JSON object
                json_start_index = pylint_output.find('{')
                if json_start_index == -1:
                     # If no '{', try '[' for list output (e.g., if no issues found)
                     json_start_index = pylint_output.find('[')
                     if json_start_index == -1:
                        # If still no JSON, it might be empty output or just text
                        if pylint_output.strip() == "":
                             return pylint_metrics, pylint_errors_counter # No issues found
                        raise json.JSONDecodeError("No JSON object/array start found in pylint output.", pylint_output, 0)

                json_string = pylint_output[json_start_index:]
                
                # Handle case where pylint outputs a list of messages directly
                # or the full object with stats.
                if json_string.startswith('['):
                    # Output is just a list of messages
                    messages = json.loads(json_string)
                    stats = {} # No stats block
                else:
                    # Output is a JSON object
                    data = json.loads(json_string)
                    messages = data.get('messages', [])
                    stats = data.get('statistics', {})

            except json.JSONDecodeError as e:
                print(f"  Error parsing pylint JSON output: {e}")
                print(f"  Pylint output (raw) was: {pylint_output[:500]}...")
                return pylint_metrics, pylint_errors_counter

            # 1. Statistics
            msg_counts = stats.get('messageTypeCount', {})
            pylint_metrics['python_pylint_fatal'] = msg_counts.get('fatal', 0)
            pylint_metrics['python_pylint_error'] = msg_counts.get('error', 0)
            pylint_metrics['python_pylint_warning'] = msg_counts.get('warning', 0)

            # 2. Message Counters
            for msg in messages:
                msg_type = msg.get('type', 'UNKNOWN')
                msg_symbol = msg.get('symbol', 'UNKNOWN')
                key = f"{msg_type}_{msg_symbol}"
                pylint_errors_counter[key] += 1
            
            return pylint_metrics, pylint_errors_counter

        except FileNotFoundError:
            print("  'pylint' command not found. Please install it ('pip install pylint') and ensure it's in your PATH.")
            return pylint_metrics, pylint_errors_counter
        except Exception as e:
            print(f"  An unexpected error occurred during pylint analysis: {e}")
            return pylint_metrics, pylint_errors_counter
        finally:
            os.chdir(original_dir)


    def analyze(self, commit_sha):
        """
        Performs all registered analyses for a given commit.
        Args:
            commit_sha (str): The commit SHA to analyze.
        Returns:
            dict: A dictionary of metric names to their values.
        """
        print(f"  Analyzing commit {commit_sha[:7]}...")
        if not self._checkout_and_clean(commit_sha):
            return {}  # Return empty dict on checkout failure

        metrics = {}
        
        # --- Lines of Code (LOC) Analysis ---
        loc = self._analyze_loc()
        if loc is not None:
            metrics['loc'] = loc
            
        # --- Cyclomatic Complexity Analysis ---
        cyclomatic_sum, over_cyclomatic_count, cognitive_sum, over_cognitive_count = self._analyze_complexity()
        if cyclomatic_sum is not None: # Check one, assume all are set
            metrics['python_flake8_cyclomatic_sum'] = cyclomatic_sum
            metrics['python_flake8_over_cyclomatic_count'] = over_cyclomatic_count
            metrics['python_flake8_cognitive_sum'] = cognitive_sum
            metrics['python_flake8_over_cognitive_count'] = over_cognitive_count
            
        # --- Security Analysis ---
        security_metrics, all_errors_counter, high_sev_errors_counter = self._analyze_security()
        # security_metrics is a dict, merge it
        metrics.update(security_metrics)
        # Store the counters for diffing later
        metrics['python_bandit_all_errors_counter'] = all_errors_counter
        metrics['python_bandit_high_sev_errors_counter'] = high_sev_errors_counter
            
        # --- Pylint Analysis ---
        pylint_metrics, pylint_errors_counter = self._analyze_pylint()
        metrics.update(pylint_metrics)
        metrics['python_pylint_errors_counter'] = pylint_errors_counter
            
        # --- Future analyses can be added here ---
        # e.g., complexity = self._analyze_complexity()
        # if complexity is not None:
        #     metrics['COMPLEXITY'] = complexity

        return metrics

class OutputProcessor:
    """Handles processing and exporting of analysis results."""
    
    def write_to_csv(self, all_results_dfs, output_file):
        """
        Consolidates a list of result DataFrames into a single CSV file.

        Args:
            all_results_dfs (list): A list of pandas DataFrames, each from a repo group.
            output_file (str): The path to the output CSV file.
        """
        if not all_results_dfs:
            print("No results to write to CSV.")
            return

        print(f"\nConsolidating results into '{output_file}'...")
        # Concatenate all dataframes into a single one
        final_df = pd.concat(all_results_dfs, ignore_index=True)

        # Drop rows where metrics could not be collected (e.g., checkout failures)
        # We check for the new delta metrics.
        metrics_to_check = ['delta_loc']
        if 'delta_python_flake8_cyclomatic_sum' in final_df.columns:
             metrics_to_check.extend(['delta_python_flake8_cyclomatic_sum',
                                      'delta_python_flake8_over_cyclomatic_count',
                                      'delta_python_flake8_cognitive_sum',
                                      'delta_python_flake8_over_cognitive_count'])
        
        # Add the new security metrics
        if 'delta_python_bandit_severity_high' in final_df.columns:
             metrics_to_check.extend(['delta_python_bandit_severity_high',
                                      'delta_python_bandit_severity_medium',
                                      'delta_python_bandit_severity_low'])
        
        # Add the new pylint metrics
        if 'delta_python_pylint_error' in final_df.columns:
             metrics_to_check.extend(['delta_python_pylint_fatal',
                                      'delta_python_pylint_error',
                                      'delta_python_pylint_warning'])
        
        final_df.dropna(subset=metrics_to_check, inplace=True)
        
        # Columns to remove from the final output
        cols_to_drop = ['body', 'title']
        # Drop columns if they exist, ignoring errors if they don't
        final_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        final_df.to_csv(output_file, index=False)
        print(f"Successfully saved consolidated results to '{output_file}'.")


def analyze_pr_group(pr_group_df, repo_full_name, repo_language):
    """
    Clones a repository and analyzes a group of its pull requests, adding
    metric data back into the DataFrame.
    Args:
        pr_group_df (pd.DataFrame): DataFrame containing the PRs for one repo.
        repo_full_name (str): The full name of the repository (e.g., 'owner/repo').
        repo_language (str): The primary language of the repository.
    Returns:
        pd.DataFrame: The input DataFrame, now updated with analysis columns.
    """
    failed_checkouts = 0
    print(f"\n--- Analyzing PRs for {repo_full_name} ---")
    repo_path = clone_repo(repo_full_name)
    if not repo_path:
        print(f"Could not clone or find repository {repo_full_name}. Aborting analysis.")
        return pr_group_df # Return the original df

    analyzer = Analyzer(repo_path, repo_language)
    
    # Prepare columns for the new data
    new_cols = [
        'base_sha', 'head_sha',
        'delta_loc',
        'delta_python_flake8_cyclomatic_sum', 'delta_python_flake8_over_cyclomatic_count',
        'delta_python_flake8_cognitive_sum', 'delta_python_flake8_over_cognitive_count',
        'delta_python_bandit_syntax_errors', 'delta_python_bandit_severity_high', 'delta_python_bandit_severity_medium', 'delta_python_bandit_severity_low',
        'new_python_bandit_top_1_error', 'new_python_bandit_top_1_error_count',
        'new_python_bandit_top_2_error', 'new_python_bandit_top_2_error_count',
        'new_python_bandit_top_3_error', 'new_python_bandit_top_3_error_count',
        'removed_python_bandit_top_1_error', 'removed_python_bandit_top_1_error_count',
        'removed_python_bandit_top_2_error', 'removed_python_bandit_top_2_error_count',
        'removed_python_bandit_top_3_error', 'removed_python_bandit_top_3_error_count',
        'new_python_bandit_top_1_high_sev_error', 'new_python_bandit_top_1_high_sev_error_count',
        'removed_python_bandit_top_1_high_sev_error', 'removed_python_bandit_top_1_high_sev_error_count',
        'delta_python_pylint_fatal', 'delta_python_pylint_error', 'delta_python_pylint_warning',
        'new_python_pylint_top_1_error', 'new_python_pylint_top_1_error_count',
        'new_python_pylint_top_2_error', 'new_python_pylint_top_2_error_count',
        'new_python_pylint_top_3_error', 'new_python_pylint_top_3_error_count',
        'removed_python_pylint_top_1_error', 'removed_python_pylint_top_1_error_count',
        'removed_python_pylint_top_2_error', 'removed_python_pylint_top_2_error_count',
        'removed_python_pylint_top_3_error', 'removed_python_pylint_top_3_error_count'
    ]

    # Initialize all new columns with pd.NA
    for col in new_cols:
        if col not in pr_group_df.columns:
            pr_group_df[col] = pd.NA

    # Set default counts to 0 and error names to N/A for this group
    string_cols = [col for col in new_cols if 'error' in col and '_count' not in col]
    count_cols = [col for col in new_cols if '_count' in col]
    
    pr_group_df[string_cols] = 'N/A'
    pr_group_df[count_cols] = 0

    for index, pr_row in pr_group_df.iterrows():
        pr_number = pr_row['number']
        print(f"\nProcessing PR #{pr_number}...")

        base_sha, head_sha = get_pr_base_and_head(repo_full_name, pr_number)
        if not (base_sha and head_sha):
            print(f"  Could not retrieve base/head SHAs for PR #{pr_number}. Skipping.")
            continue

        pr_group_df.loc[index, 'base_sha'] = base_sha
        pr_group_df.loc[index, 'head_sha'] = head_sha
        
        # Analyze base commit
        base_metrics = analyzer.analyze(base_sha)
        # Analyze head commit
        head_metrics = analyzer.analyze(head_sha)
        
        if not base_metrics or not head_metrics:
            failed_checkouts += 1
            print(f"  No metrics collected for PR #{pr_number} due to checkout/analysis failure.")
            continue
        
        # --- Calculate Deltas for simple metrics ---
        delta_metrics = {}
        simple_metrics_to_delta = [
            'loc', 
            'python_flake8_cyclomatic_sum', 'python_flake8_over_cyclomatic_count',
            'python_flake8_cognitive_sum', 'python_flake8_over_cognitive_count',
            'python_bandit_syntax_errors', 'python_bandit_severity_high', 'python_bandit_severity_medium', 'python_bandit_severity_low',
            'python_pylint_fatal', 'python_pylint_error', 'python_pylint_warning'
        ]
        
        for metric in simple_metrics_to_delta:
            base_val = base_metrics.get(metric)
            head_val = head_metrics.get(metric)
            
            if base_val is not None and head_val is not None:
                # Store the delta: head - base
                delta_metrics[f'delta_{metric}'] = head_val - base_val
            else:
                # Set delta to NA if either value is missing (e.g., from failed locstats)
                delta_metrics[f'delta_{metric}'] = pd.NA

        for metric, value in delta_metrics.items():
            pr_group_df.loc[index, metric] = value

        # --- Calculate Error Diffs (New vs. Removed) ---
        # Bandit
        base_bandit_all_errors = base_metrics.get('python_bandit_all_errors_counter', Counter())
        head_bandit_all_errors = head_metrics.get('python_bandit_all_errors_counter', Counter())
        base_bandit_high_errors = base_metrics.get('python_bandit_high_sev_errors_counter', Counter())
        head_bandit_high_errors = head_metrics.get('python_bandit_high_sev_errors_counter', Counter())

        new_bandit_all_errors = head_bandit_all_errors - base_bandit_all_errors
        removed_bandit_all_errors = base_bandit_all_errors - head_bandit_all_errors
        new_bandit_high_errors = head_bandit_high_errors - base_bandit_high_errors
        removed_bandit_high_errors = base_bandit_high_errors - head_bandit_high_errors

        # Pylint
        base_pylint_errors = base_metrics.get('python_pylint_errors_counter', Counter())
        head_pylint_errors = head_metrics.get('python_pylint_errors_counter', Counter())

        new_pylint_errors = head_pylint_errors - base_pylint_errors
        removed_pylint_errors = base_pylint_errors - head_pylint_errors


        # --- Store Top 3 New/Removed Bandit All Errors ---
        top_3_new_bandit = new_bandit_all_errors.most_common(3)
        for i, (name, count) in enumerate(top_3_new_bandit, 1):
            if count > 0: # Only store if there's a positive count
                pr_group_df.loc[index, f'new_python_bandit_top_{i}_error'] = name
                pr_group_df.loc[index, f'new_python_bandit_top_{i}_error_count'] = count
        
        top_3_removed_bandit = removed_bandit_all_errors.most_common(3)
        for i, (name, count) in enumerate(top_3_removed_bandit, 1):
            if count > 0: # Only store if there's a positive count
                pr_group_df.loc[index, f'removed_python_bandit_top_{i}_error'] = name
                pr_group_df.loc[index, f'removed_python_bandit_top_{i}_error_count'] = count
        
        # --- Store Top 1 New/Removed Bandit High Severity Errors ---
        top_1_new_bandit_high = new_bandit_high_errors.most_common(1)
        if top_1_new_bandit_high and top_1_new_bandit_high[0][1] > 0:
            pr_group_df.loc[index, 'new_python_bandit_top_1_high_sev_error'] = top_1_new_bandit_high[0][0]
            pr_group_df.loc[index, 'new_python_bandit_top_1_high_sev_error_count'] = top_1_new_bandit_high[0][1]

        top_1_removed_bandit_high = removed_bandit_high_errors.most_common(1)
        if top_1_removed_bandit_high and top_1_removed_bandit_high[0][1] > 0:
            pr_group_df.loc[index, 'removed_python_bandit_top_1_high_sev_error'] = top_1_removed_bandit_high[0][0]
            pr_group_df.loc[index, 'removed_python_bandit_top_1_high_sev_error_count'] = top_1_removed_bandit_high[0][1]

        # --- Store Top 3 New/Removed Pylint Errors ---
        top_3_new_pylint = new_pylint_errors.most_common(3)
        for i, (name, count) in enumerate(top_3_new_pylint, 1):
            if count > 0: # Only store if there's a positive count
                pr_group_df.loc[index, f'new_python_pylint_top_{i}_error'] = name
                pr_group_df.loc[index, f'new_python_pylint_top_{i}_error_count'] = count
        
        top_3_removed_pylint = removed_pylint_errors.most_common(3)
        for i, (name, count) in enumerate(top_3_removed_pylint, 1):
            if count > 0: # Only store if there's a positive count
                pr_group_df.loc[index, f'removed_python_pylint_top_{i}_error'] = name
                pr_group_df.loc[index, f'removed_python_pylint_top_{i}_error_count'] = count


    print(f"\n--- Finished analysis for {repo_full_name} ---")
    print(f"Total PRs processed: {len(pr_group_df)}")
    print(f"Failed checkouts/analyses: {failed_checkouts}")
    return pr_group_df

def main():
    """
    Loads the AIDev dataset, processes pull requests from the first
    repository found, analyzes them, and saves the results to a CSV file.
    """
    dataset_repo_id = "hao-li/AIDev"
    target_languages = ['Python', 'TypeScript', 'JavaScript']

    print(f"Loading repository data from '{dataset_repo_id}'...")
    try:
        repo_dataset = load_dataset(dataset_repo_id, 'all_repository', split='train')
        pr_dataset = load_dataset(dataset_repo_id, 'all_pull_request', split='train')
        repo_df = pd.DataFrame(repo_dataset)
        pr_df = pd.DataFrame(pr_dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Filtering for repositories in: {target_languages}")
    filtered_repos = repo_df[repo_df['language'].isin(target_languages)]
    if filtered_repos.empty:
        print("No repositories found for the specified languages.")
        return

    filtered_repo_ids = filtered_repos['id'].unique()
    print(f"Found {len(filtered_repo_ids)} repositories matching language criteria.")

    filtered_prs = pr_df[pr_df['repo_id'].isin(filtered_repo_ids)]
    if filtered_prs.empty:
        print("No pull requests found for the filtered repositories.")
        return

    grouped_prs = filtered_prs.groupby('repo_id')
    if grouped_prs.ngroups == 0:
        print("No pull requests found to group.")
        return

    print(f"Found {grouped_prs.ngroups} repositories with relevant pull requests.")

    # --- Process repository groups ---
    all_results_dfs = []
    
    # Create a mapping of repo_id -> info
    repo_info_map = filtered_repos.set_index('id').to_dict('index')
    
    # Get all repo_ids that have PRs
    repo_ids_with_prs = set(grouped_prs.groups.keys())

    processed_langs = set()

    print("---")
    print(f"Attempting to find and process one clonable repo for each language: {target_languages}")
    print("---")

    # Iterate through all repositories that match our language criteria
    # We iterate through the DataFrame to maintain some order
    for repo_id, repo_info in repo_info_map.items():
        lang = repo_info['language']
        
        # If we already processed this language, or this repo has no PRs, skip
        if lang not in target_languages or lang in processed_langs or repo_id not in repo_ids_with_prs:
            continue
            
        repo_full_name = repo_info['full_name']
        
        print(f"\nAttempting to process first available repo for language: {lang}")
        print(f"Trying repo: {repo_full_name} (Repo ID: {repo_id})")
        
        # Try to clone the repo. clone_repo will print success/failure
        repo_path = clone_repo(repo_full_name)
        
        # If cloning fails, repo_path will be None, and we continue to the next repo
        if repo_path:
            print(f"Successfully cloned '{repo_full_name}'. Proceeding with analysis.")
            
            # Get the DataFrame for this group of PRs
            pr_group_df = grouped_prs.get_group(repo_id)

            # Analyze the group of PRs and get the results as an updated DataFrame
            # Use .copy() to avoid SettingWithCopyWarning
            results_df = analyze_pr_group(pr_group_df.copy(), repo_full_name, lang)
            
            # Add the language to the results DataFrame
            results_df['repo_language'] = lang
            
            all_results_dfs.append(results_df)
            
            # Mark this language as processed and move to the next
            processed_langs.add(lang)
            
            # Check if we have processed all target languages
            if len(processed_langs) == len(target_languages):
                print("\nProcessed one repository for all target languages.")
                break
        else:
            # clone_repo already printed the error, just note we're skipping
            print(f"Skipping repo '{repo_full_name}', will try next available repo for {lang}.")


    # --- Process and save the final results ---
    if all_results_dfs:
        processor = OutputProcessor()
        processor.write_to_csv(all_results_dfs, 'analysis_results.csv')
    else:
        print("\nNo data was processed.")


if __name__ == "__main__":
    # Make sure you have the required libraries installed:
    # pip install datasets pandas requests locstats
    main()

