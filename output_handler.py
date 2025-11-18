import pandas as pd
import os
import glob

class OutputHandler:
    def __init__(self, directory="."):
        self.directory = directory

    """
    Handles processing and exporting of analysis results to a CSV file.
    """
    
    def get_processed_repos(self):
        """
        Scans the output directory for CSV files and returns a set of
        repository full names that have already been processed.
        
        Returns:
            set: A set of strings, where each string is a 'repo_full_name'
                 (e.g., 'owner/repo') found in existing CSVs.
        """
        print(f"Scanning '{self.directory}' for previously processed repositories...")
        search_pattern = os.path.join(self.directory, "*_analysis_results.csv")
        csv_files = glob.glob(search_pattern)
        
        processed_repos = set()
        
        if not csv_files:
            print("No existing analysis files found.")
            return processed_repos

        for f in csv_files:
            try:
                # Only read the 'repo_full_name' column to save memory
                df = pd.read_csv(f, usecols=['repo_full_name'])
                if not df.empty and 'repo_full_name' in df.columns:
                    repos_in_file = df['repo_full_name'].unique()
                    processed_repos.update(repos_in_file)
            except Exception as e:
                print(f"Could not read repo name from '{f}': {e}")
                
        return processed_repos

    def process_and_save_csv(self, results_df, output_file):
        """
        Processes a single repository's results DataFrame and saves it to a CSV file.

        Args:
            results_df (pd.DataFrame): The DataFrame of results for one repository.
            output_file (str): The path to the output CSV file.
        """
        if results_df.empty:
            print(f"No results to write to '{output_file}'.")
            return

        print(f"\nProcessing results for '{output_file}'...")
        
        final_df = results_df.copy()

        # Define metrics to check for NAs. If these are NA, the analysis failed.
        metrics_to_check = ['delta_loc']
        if 'delta_python_flake8_cyclomatic_sum' in final_df.columns:
            metrics_to_check.extend([
                'delta_python_flake8_cyclomatic_sum',
                'delta_python_flake8_over_cyclomatic_count',
                'delta_python_flake8_cognitive_sum',
                'delta_python_flake8_over_cognitive_count'
            ])
        
        if 'delta_python_bandit_severity_high' in final_df.columns:
            metrics_to_check.extend([
                'delta_python_bandit_severity_high',
                'delta_python_bandit_severity_medium',
                'delta_python_bandit_severity_low'
            ])
        
        if 'delta_python_pylint_error' in final_df.columns:
            metrics_to_check.extend([
                'delta_python_pylint_fatal',
                'delta_python_pylint_error',
                'delta_python_pylint_warning'
            ])

        for column in metrics_to_check:
            if column not in final_df.columns:
                print(f"Warning: Expected metric column '{column}' not found in results DataFrame.")

        initial_rows = len(final_df)
        final_df.dropna(subset=metrics_to_check, inplace=True)
        dropped_rows = initial_rows - len(final_df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NA metrics.")

        if final_df.empty:
            print(f"No valid data remaining after processing for '{output_file}'.")
            return

        # Columns to remove from the final output
        cols_to_drop = ['body', 'title']
        final_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        f = open(f"{self.directory}/{output_file}","w",newline="")

        final_df.to_csv(f, index=False)
        f.flush()
        f.close()
        print(f"Successfully saved processed results to '{output_file}'.")