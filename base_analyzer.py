import os
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """
    Abstract Base Class for a language-specific code analyzer.
    
    It defines the interface that all concrete analyzers must implement.
    It provides method to build the metrics dictionary
    """
    
    def __init__(self, repo_path, git_handler):
        """
        Initializes the analyzer.

        Args:
            repo_path (str): The local path to the cloned repository.
            git_handler (GitHandler): An instance of GitHandler to perform checkouts.
        """
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
        self.repo_path = repo_path
        self.git_handler = git_handler
        self.language = self.get_language() # Will be set by subclass
        self._metrics = {}
        print(f"Initialized {self.__class__.__name__} for language '{self.language}' at '{self.repo_path}'")

    @abstractmethod
    def get_language(self):
        """
        Subclass must implement this to return its language (e.g., 'python').
        """
        pass

    @abstractmethod
    def analyze_loc(self):
        """
        Analyzes Lines of Code (LOC) and calls _set_loc.
        """
        pass

    @abstractmethod
    def analyze_complexity(self):
        """
        Analyzes cyclomatic and cognitive complexity.
        Calls _set_cyclomatic_complexity and _set_cognitive_complexity.
        """
        pass

    @abstractmethod
    def analyze_security_faults(self):
        """
        Analyzes for security vulnerabilities.
        Calls _set_security_metrics and _set_security_counters.
        """
        pass

    @abstractmethod
    def analyze_general_faults(self):
        """
        Analyzes for general code smells, errors, and warnings.
        Calls _set_general_fault_metrics and _set_general_fault_counters.
        """
        pass

    @abstractmethod
    def analyze_pr_deltas(self, base_sha, head_sha):
        """
        Orchestrates the full analysis of a PR.
        This method is responsible for:
        1. Checking out the base_sha
        2. Calling `build_metric_dict()` to get base metrics
        3. Checking out the head_sha
        4. Calling `build_metric_dict()` to get head metrics
        5. Calculating all deltas (simple subtraction, counter diffs)
        6. Returning a single dictionary of delta metrics.

        Args:
            base_sha (str): The commit SHA of the base branch.
            head_sha (str): The commit SHA of the head branch.

        Returns:
            dict: A dictionary of all delta metrics, or None on failure.
        """
        pass

    def build_metric_dict(self):
        """
        Orchestrates the full analysis for the *currently checked out* commit.
        It runs all individual analysis methods which populate self._metrics.

        Returns:
            dict: A dictionary of all aggregated metrics for the current commit.
        """
        print(f"  Building metric dict for {self.__class__.__name__}...")
        self._metrics = {} 
        
        self.analyze_loc()
        self.analyze_complexity()
        self.analyze_security_faults()
        self.analyze_general_faults()
        
        return self._metrics.copy() 

    # --- Standardized Metric Setters ---

    def _set_loc(self, value):
        """Sets the Lines of Code metric."""
        self._metrics['loc'] = value

    def _set_cyclomatic_complexity(self, sum_val, over_count_val):
        """Sets cyclomatic complexity metrics."""
        self._metrics['cyclomatic_sum'] = sum_val
        self._metrics['over_cyclomatic_count'] = over_count_val

    def _set_cognitive_complexity(self, sum_val, over_count_val):
        """Sets cognitive complexity metrics."""
        self._metrics['cognitive_sum'] = sum_val
        self._metrics['over_cognitive_count'] = over_count_val

    def _set_security_metrics(self, syntax_errors, high, medium, low):
        """Sets security fault count metrics."""
        self._metrics['security_syntax_errors'] = syntax_errors
        self._metrics['security_severity_high'] = high
        self._metrics['security_severity_medium'] = medium
        self._metrics['security_severity_low'] = low

    def _set_security_counters(self, all_errors_counter, high_sev_errors_counter):
        """Sets security fault Counter objects."""
        self._metrics['security_all_errors_counter'] = all_errors_counter
        self._metrics['security_high_sev_errors_counter'] = high_sev_errors_counter

    def _set_general_fault_metrics(self, fatal, error, warning):
        """Sets general fault count metrics."""
        self._metrics['general_fatal'] = fatal
        self._metrics['general_error'] = error
        self._metrics['general_warning'] = warning

    def _set_general_fault_counters(self, errors_counter):
        """Sets general fault Counter object."""
        self._metrics['general_errors_counter'] = errors_counter

    def _populate_top_n_deltas(self, delta_dict, prefix, counter, n):
        """
        Helper function to populate a delta dictionary with the top N
        items from a Counter object.

        Args:
            delta_dict (dict): The dictionary to populate.
            prefix (str): The prefix for the new keys (e.g., 'new_security').
            counter (Counter): The Counter object (e.g., new_sec_all).
            n (int): The number of top items to get.
        """
        # Initialize default values
        for i in range(1, n + 1):
            delta_dict[f'{prefix}_top_{i}_error'] = 'N/A'
            delta_dict[f'{prefix}_top_{i}_error_count'] = 0
            
        # Populate with most common
        for i, (name, count) in enumerate(counter.most_common(n), 1):
            if count > 0:
                delta_dict[f'{prefix}_top_{i}_error'] = name
                delta_dict[f'{prefix}_top_{i}_error_count'] = count
