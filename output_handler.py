import pandas as pd

class OutputHandler:
    def __init__(self, directory="."):
        self.directory = directory

    """
    Handles processing and exporting of analysis results to a CSV file.
    """
    
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

        # Drop rows where core metrics could not be collected
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

        final_df.to_csv(f"{self.directory}/{output_file}", index=False)
        print(f"Successfully saved processed results to '{output_file}'.")
