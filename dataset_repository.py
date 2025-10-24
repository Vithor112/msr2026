import pandas as pd
from datasets import load_dataset

class DatasetRepository:
    """
    Handles all operations related to fetching data from the 
    Hugging Face dataset.
    """
    def __init__(self, dataset_repo_id="hao-li/AIDev"):
        """
        Initializes the repository with the dataset ID.

        Args:
            dataset_repo_id (str): The ID of the Hugging Face dataset 
                                   (e.g., 'hao-li/AIDev').
        """
        self.dataset_repo_id = dataset_repo_id
        print(f"Initializing DatasetRepository for '{dataset_repo_id}'")

    def get_repositories_df(self):
        """
        Loads and returns the 'all_repository' split as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame of the repositories.
        """
        try:
            repo_dataset = load_dataset(self.dataset_repo_id, 'all_repository', split='train')
            return pd.DataFrame(repo_dataset)
        except Exception as e:
            print(f"Error loading 'all_repository' dataset: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

    def get_pull_requests_df(self):
        """
        Loads and returns the 'all_pull_request' split as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame of the pull requests.
        """
        try:
            pr_dataset = load_dataset(self.dataset_repo_id, 'all_pull_request', split='train')
            return pd.DataFrame(pr_dataset)
        except Exception as e:
            print(f"Error loading 'all_pull_request' dataset: {e}")
            return pd.DataFrame() # Return empty DataFrame on error
