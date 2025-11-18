import os
import pandas as pd

from dataset_repository import DatasetRepository
from git_handler import GitHandler
from python_analyzer import PythonAnalyzer
from output_handler import OutputHandler

ANALYZER_REGISTRY = {
    "Python": PythonAnalyzer
    }

def analyze_pr_group(pr_group_df, repo_full_name, repo_language, analyzer_class, git_handler, repo_path):
    """
    Analyzes a group of pull requests for a single repository.

    The analyzer class is now responsible for all delta calculations.
    This function just retrieves the delta dict and updates the DataFrame.

    Args:
        pr_group_df (pd.DataFrame): DataFrame containing the PRs for one repo.
        repo_full_name (str): The full name of the repository (e.g., 'owner/repo').
        repo_language (str): The primary language of the repository.
        analyzer_class (BaseAnalyzer): The concrete analyzer class to use (e.g., PythonAnalyzer).
        git_handler (GitHandler): The initialized GitHandler instance.
        repo_path (str): The local path to the cloned repository.

    Returns:
        pd.DataFrame: The input DataFrame, updated with analysis columns.
    """
    failed_analyses = 0
    print(f"\n--- Analyzing PRs for {repo_full_name} ({repo_language}) ---")

    analyzer = analyzer_class(repo_path, git_handler)

    counter = 0

    if 'base_sha' not in pr_group_df.columns:
        pr_group_df['base_sha'] = pd.NA
    if 'head_sha' not in pr_group_df.columns:
        pr_group_df['head_sha'] = pd.NA

    for index, pr_row in pr_group_df.iterrows():
        pr_number = pr_row['number']
        print(f"\nProcessing PR #{pr_number}...")

        base_sha, head_sha = git_handler.get_pr_base_and_head(repo_full_name, pr_number)
        if not (base_sha and head_sha):
            print(f"  Could not retrieve base/head SHAs for PR #{pr_number}. Skipping.")
            failed_analyses += 1
            continue

        pr_group_df.loc[index, 'base_sha'] = base_sha
        pr_group_df.loc[index, 'head_sha'] = head_sha

        delta_metrics = analyzer.analyze_pr_deltas(base_sha, head_sha)

        if not delta_metrics:
            failed_analyses += 1
            print(f"  No metrics collected for PR #{pr_number} due to checkout/analysis failure.")
            continue

        for key, value in delta_metrics.items():
            if key not in pr_group_df.columns:
                pr_group_df[key] = pd.NA
            pr_group_df.loc[index, key] = value
        counter += 1
    print(f"\n--- Finished analysis for {repo_full_name} ---")
    print(f"Total PRs processed: {len(pr_group_df)}")
    print(f"Failed analyses (no SHAs or checkout/analysis error): {failed_analyses}")
    return pr_group_df, failed_analyses

def main():
    """
    Main orchestration function.
    - Loads data
    - Filters repos based on registered analyzers
    - Loops through repos, clones, analyzes, and saves results
    """


    dataset_repo = DatasetRepository("hao-li/AIDev")
    directory_git = 'repositories'
    if not os.path.exists(directory_git):
        os.makedirs(directory_git)
    git_handler = GitHandler(github_token, directory_git)
    directory = 'output'
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_handler = OutputHandler(directory)

    processed_repos = output_handler.get_processed_repos()
    if processed_repos:
        print(f"Found {len(processed_repos)} already processed repositories. They will be skipped.")
        print(f"Processed list: {processed_repos}")

    target_languages = list(ANALYZER_REGISTRY.keys())
    if not target_languages:
        print("No analyzers are registered in ANALYZER_REGISTRY. Exiting.")
        return

    print(f"Registered analyzers for: {target_languages}")

    print(f"\nLoading repository data...")
    repo_df = dataset_repo.get_repositories_df()
    pr_df = dataset_repo.get_pull_requests_df()

    if repo_df.empty or pr_df.empty:
        print("Failed to load initial data. Exiting.")
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
    print(f"Found {grouped_prs.ngroups} repositories with relevant pull requests.")

    repo_info_map = filtered_repos.set_index('id').to_dict('index')
    repo_ids_with_prs = set(grouped_prs.groups.keys())

    print("---")
    print(f"Attempting to find and process one clonable repo for each language: {target_languages}")
    print("---")
    for repo_id, repo_info in repo_info_map.items():
        lang = repo_info['language']

        if repo_id not in repo_ids_with_prs:
            continue

        repo_full_name = repo_info['full_name']

        if repo_full_name in processed_repos:
            print(f"\nSkipping '{repo_full_name}' as it appears in existing CSV files.")
            continue

        print(f"\nAttempting to process first available repo for language: {lang}")
        print(f"Trying repo: {repo_full_name} (Repo ID: {repo_id})")

        repo_path = git_handler.clone_repo(repo_full_name)

        if repo_path:
            print(f"Successfully cloned '{repo_full_name}'. Proceeding with analysis.")

            analyzer_class = ANALYZER_REGISTRY[lang]
            pr_group_df = grouped_prs.get_group(repo_id)
            results_df, failed_analy = analyze_pr_group(
                pr_group_df.copy(),
                repo_full_name,
                lang,
                analyzer_class,
                git_handler,
                repo_path
            )

            if (len(results_df) - failed_analy) == 0:
                print(f"All PR analyses failed for '{repo_full_name}'. Skipping saving results.")
                continue

            results_df['repo_language'] = lang
            results_df['repo_full_name'] = repo_full_name

            repo_name_safe = repo_full_name.replace('/', '_')
            output_file = f"{repo_name_safe}_analysis_results.csv"
            output_handler.process_and_save_csv(results_df, output_file)
        else:
            print(f"Skipping repo '{repo_full_name}', will try next available repo for {lang}.")

    print("\n--- Main process finished ---")

if __name__ == "__main__":
    main()