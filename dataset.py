import json
import re
import random
from collections import Counter
from datasets import load_dataset

# --- Configuration ---
REPO_ID = "hao-li/AIDev"
OUTPUT_FILE = "pr_analysis_data.json"
PR_SAMPLE_SIZE = 100
# Common English stop words
STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn", "pr", "pull", "request"
])

def clean_text(text):
    """Cleans and tokenizes text for word frequency analysis."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"```[\s\S]*?```", "", text) # Remove code blocks
    text = re.sub(r"`[^`]*?`", "", text) # Remove inline code
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    tokens = text.split()
    return [word for word in tokens if word.isalpha() and word not in STOP_WORDS]

def is_human_commenter(username):
    """Checks if a username is likely a human."""
    if not username:
        return False
    username_lower = username.lower()
    return 'bot' not in username_lower and 'assistant' not in username_lower and 'github' not in username_lower and username_lower != 'copilot'

def main():
    """Main function to process datasets and generate analysis file."""
    print("Loading datasets...")
    try:
        pr_dataset = load_dataset(REPO_ID, 'all_pull_request', trust_remote_code=True)
        comments_dataset = load_dataset(REPO_ID, 'pr_comments', trust_remote_code=True)
        reviews_dataset = load_dataset(REPO_ID, 'pr_reviews', trust_remote_code=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    print("Identifying rejected pull requests...")
    rejected_prs = {
        pr['id']: pr for pr in pr_dataset['train']
        if pr['closed_at'] is not None and pr['merged_at'] is None
    }
    rejected_pr_ids = set(rejected_prs.keys())
    print(f"Found {len(rejected_pr_ids)} rejected PRs.")

    print("Grouping all comments and reviews by pull request...")
    interactions_by_pr = {}
    for comment in comments_dataset['train']:
        pr_id = comment.get('pr_id')
        if pr_id in rejected_pr_ids:
            interactions_by_pr.setdefault(pr_id, []).append({
                "type": "comment", "user": comment.get('user'), "body": comment.get('body'), "created_at": comment.get('created_at')
            })
    for review in reviews_dataset['train']:
        pr_id = review.get('pr_id')
        if pr_id in rejected_pr_ids and review.get('body'):
            interactions_by_pr.setdefault(pr_id, []).append({
                "type": "review", "user": review.get('user'), "body": review.get('body'), "created_at": review.get('submitted_at')
            })
    
    print("Analyzing metrics and identifying PRs with human feedback...")
    rejected_without_human_interaction = 0
    rejected_without_human_review = 0
    candidate_pr_ids_for_sampling = []

    for pr_id in rejected_pr_ids:
        interactions = interactions_by_pr.get(pr_id, [])
        human_interactions = [i for i in interactions if is_human_commenter(i.get('user'))]
        human_reviews = [i for i in human_interactions if i['type'] == 'review']

        if not human_interactions:
            rejected_without_human_interaction += 1
        else:
            candidate_pr_ids_for_sampling.append(pr_id)
        
        if not human_reviews:
            rejected_without_human_review += 1
    
    print("\n--- Rejection Metrics ---")
    print(f"Total Rejected PRs: {len(rejected_pr_ids)}")
    print(f"PRs rejected without any human comments or reviews: {rejected_without_human_interaction}")
    print(f"PRs rejected without any human reviews: {rejected_without_human_review}")
    print("-----------------------\n")

    print(f"Found {len(candidate_pr_ids_for_sampling)} PRs with human feedback for sampling.")
    
    print("Randomly sampling PRs for visualization...")
    random.shuffle(candidate_pr_ids_for_sampling)
    sampled_pr_ids = candidate_pr_ids_for_sampling[:PR_SAMPLE_SIZE]

    rejected_prs_with_details = []
    for pr_id in sampled_pr_ids:
        pr_info = rejected_prs.get(pr_id, {})
        all_interactions = interactions_by_pr.get(pr_id, [])
        sorted_interactions = sorted(all_interactions, key=lambda c: c.get('created_at', '') or '')
        
        rejected_prs_with_details.append({
            "repo_name": '/'.join(pr_info.get('repo_url', '').split('/')[-2:]),
            "repo_url": pr_info.get('repo_url'),
            "pr_id": pr_id,
            "pr_title": pr_info.get('title'),
            "pr_body": pr_info.get('body'),
            "agent": pr_info.get('agent'),
            "comments": [{
                "author": c.get('user'), "body": c.get('body'), "created_at": c.get('created_at')
            } for c in sorted_interactions]
        })

    print(f"Sampled {len(rejected_prs_with_details)} rejected PRs with full details.")

    print("Analyzing word frequency in human rejection comments...")
    all_human_comment_words = []
    for pr_data in rejected_prs_with_details:
        for comment in pr_data['comments']:
            if is_human_commenter(comment.get('author')):
                all_human_comment_words.extend(clean_text(comment['body']))

    word_counts = Counter(all_human_comment_words)
    word_cloud_data = [{"text": word, "size": count} for word, count in word_counts.most_common(100)]

    repo_rejection_counts = Counter()
    for pr_id in rejected_pr_ids:
        repo_url = rejected_prs[pr_id].get('repo_url')
        if repo_url:
            short_name = '/'.join(repo_url.split('/')[-2:])
            repo_rejection_counts[short_name] += 1
    
    top_repos = repo_rejection_counts.most_common(20)
    repo_chart_data = [{"repo": repo, "count": count} for repo, count in top_repos]

    output_data = {
        "word_cloud_data": word_cloud_data,
        "repo_chart_data": repo_chart_data,
        "rejected_prs_details": rejected_prs_with_details,
        "metrics": {
            "total_rejected": len(rejected_pr_ids),
            "rejected_without_human_interaction": rejected_without_human_interaction,
            "rejected_without_human_review": rejected_without_human_review
        }
    }

    print(f"Saving analysis to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()

