import os
import subprocess
import requests

class GitHandler:
    """
    Handles all Git operations, such as cloning, checking out commits,
    and interacting with the GitHub API.
    """
    def __init__(self, github_token=None, directory="."):
        """
        Initializes the GitHandler.

        Args:
            github_token (str, optional): A GitHub API token to increase 
                                          rate limits.
        """
        self.github_token = github_token
        self.directory = directory
        self.headers = {}
        if self.github_token:
            self.headers['Authorization'] = f"token {self.github_token}"
            print("GitHandler initialized with a GitHub token.")
        else:
            print("GitHandler initialized without a GitHub token. API calls may be rate-limited.")

    def get_pr_base_and_head(self, repo_full_name, pr_number):
        """
        Fetches the base and head commit sha for a given pull request.

        Args:
            repo_full_name (str): The full name of the repository (e.g., 'owner/repo').
            pr_number (int): The number of the pull request.

        Returns:
            tuple: (base_sha, head_sha) or (None, None) on failure.
        """
        api_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
        print(f"Fetching PR details from: {api_url}")
        
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            
            pr_data = response.json()
            base_sha = pr_data['base']['sha']
            head_sha = pr_data['head']['sha']
            return base_sha, head_sha
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching PR data from GitHub API: {e}")
            if not self.github_token:
                print("  This could be due to rate limiting. You can set a GITHUB_TOKEN environment variable.")
            return None, None
        except KeyError:
            print("  Error: 'base' or 'head' sha not found in API response.")
            return None, None

    def clone_repo(self, repo_full_name):
        """
        Checks if a repository is accessible, then clones it if it doesn't already exist.

        Args:
            repo_full_name (str): The full name of the repository (e.g., 'owner/repo').

        Returns:
            str: The local path to the cloned repo (e.g., './repo_name'), or None on failure.
        """
        clone_url = f"https://github.com/{repo_full_name}.git"
        repo_name = repo_full_name.split('/')[-1]

        print(f"Verifying accessibility of '{repo_full_name}'...")
        original_dir = os.getcwd()
        try:
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
        path = f"{self.directory}/{repo_name}"
        if not os.path.exists(path):
            print(f"Cloning '{repo_full_name}' into './{path}'...")
            try:
                env = os.environ.copy()
                env['GIT_TERMINAL_PROMPT'] = '0'
                os.chdir(self.directory)
                subprocess.run(['git', 'clone', clone_url], check=True, capture_output=True, text=True, env=env)
                print("Repository cloned successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone repository: {e.stderr}")
                return None
            finally:
                os.chdir(original_dir)
        else:
            print(f"Repository '{repo_name}' already exists. Skipping clone.")
        
        return path

    def checkout_and_clean(self, repo_path, commit_sha):
        """
        Checks out a specific commit and cleans the working directory.

        Args:
            repo_path (str): The local path to the cloned repository.
            commit_sha (str): The commit SHA to checkout.

        Returns:
            bool: True on success, False on failure.
        """
        original_dir = os.getcwd()
        try:
            os.chdir(repo_path)
            subprocess.run(['git', 'fetch', 'origin', commit_sha], check=True, capture_output=True, text=True)
            subprocess.run(['git', 'checkout', commit_sha, '--force'], check=True, capture_output=True, text=True)
            subprocess.run(['git', 'clean', '-fd'], check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Error checking out commit {commit_sha[:7]}: {e.stderr.strip()}")
            return False
        finally:
            os.chdir(original_dir)
