import time
from datetime import datetime
from collections import defaultdict
import json
import os

class MetricsTracker:
    """
    Tracks system performance metrics across queries.
    """
    
    def __init__(self, save_path="metrics_data.json"):
        self.save_path = save_path
        self.metrics = {
            "total_queries": 0,
            "rag_success": 0,
            "web_search_fallback": 0,
            "trusted_search_used": 0,
            "general_search_used": 0,
            "complexity_distribution": {
                "simple": 0,
                "moderate": 0,
                "complex": 0
            },
            "domain_usage": {
                "medical": 0,
                "islamic": 0,
                "insurance": 0
            },
            "response_times": [],
            "worker_contributions": {
                "dense_semantic": 0,
                "bm25_keyword": 0
            },
            "validation_stats": {
                "valid": 0,
                "invalid": 0,
                "skipped": 0
            },
            "query_history": []  # Store recent queries for analysis
        }
        
        # Load existing metrics if available
        self.load_metrics()
    
    def start_query(self):
        """Start timing a query."""
        return time.time()
    
    def end_query(self, start_time):
        """Calculate and store query response time."""
        response_time = time.time() - start_time
        self.metrics["response_times"].append(response_time)
        return response_time
    
    def log_query(self, query, domain, source, complexity=None, 
                  validation=None, response_time=None, answer_preview=None):
        """
        Log a complete query with all its metadata.
        
        Args:
            query (str): User's query
            domain (str): Domain (medical, islamic, insurance)
            source (str): Where answer came from (RAG, WebSearch, etc.)
            complexity (dict): Complexity analysis result
            validation (tuple): (is_valid, reason)
            response_time (float): Time taken in seconds
            answer_preview (str): First 100 chars of answer
        """
        self.metrics["total_queries"] += 1
        
        # Track domain usage
        if domain in self.metrics["domain_usage"]:
            self.metrics["domain_usage"][domain] += 1
        
        # Track source usage
        if "RAG" in source or "Database" in source:
            self.metrics["rag_success"] += 1
        elif "Trusted" in source:
            self.metrics["trusted_search_used"] += 1
            self.metrics["web_search_fallback"] += 1
        elif "Etiqa" in source:
            self.metrics["web_search_fallback"] += 1
        elif "Web" in source or "Search" in source:
            self.metrics["general_search_used"] += 1
            self.metrics["web_search_fallback"] += 1
        
        # Track complexity distribution
        if complexity and "complexity" in complexity:
            comp_level = complexity["complexity"]
            if comp_level in self.metrics["complexity_distribution"]:
                self.metrics["complexity_distribution"][comp_level] += 1
        
        # Track validation
        if validation:
            is_valid, reason = validation
            if "skip" in reason.lower():
                self.metrics["validation_stats"]["skipped"] += 1
            elif is_valid:
                self.metrics["validation_stats"]["valid"] += 1
            else:
                self.metrics["validation_stats"]["invalid"] += 1
        
        # Store query history (last 50 queries)
        query_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate long queries
            "domain": domain,
            "source": source,
            "complexity": complexity.get("complexity") if complexity else None,
            "k_used": complexity.get("k") if complexity else None,
            "response_time": round(response_time, 2) if response_time else None,
            "validated": is_valid if validation else None,
            "answer_preview": answer_preview[:100] if answer_preview else None
        }
        
        self.metrics["query_history"].append(query_record)
        
        # Keep only last 50 queries
        if len(self.metrics["query_history"]) > 50:
            self.metrics["query_history"] = self.metrics["query_history"][-50:]
        
        # Auto-save after each query
        self.save_metrics()
    
    def log_worker_contribution(self, worker_stats):
        """
        Log which swarm workers contributed to the final answer.
        
        Args:
            worker_stats (dict): e.g., {"dense_semantic": 5, "bm25_keyword": 3}
        """
        for worker, count in worker_stats.items():
            if worker in self.metrics["worker_contributions"]:
                self.metrics["worker_contributions"][worker] += count
    
    def get_stats(self):
        """Get current statistics."""
        total = self.metrics["total_queries"]
        
        if total == 0:
            return {
                "total_queries": 0,
                "rag_success_rate": 0,
                "web_search_rate": 0,
                "avg_response_time": 0,
                "complexity_distribution": self.metrics["complexity_distribution"],
                "domain_usage": self.metrics["domain_usage"]
            }
        
        # Calculate averages and percentages
        avg_response_time = (
            sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            if self.metrics["response_times"] else 0
        )
        
        stats = {
            "total_queries": total,
            "rag_success_rate": round((self.metrics["rag_success"] / total) * 100, 1),
            "web_search_rate": round((self.metrics["web_search_fallback"] / total) * 100, 1),
            "trusted_search_rate": round((self.metrics["trusted_search_used"] / total) * 100, 1),
            "general_search_rate": round((self.metrics["general_search_used"] / total) * 100, 1),
            "avg_response_time": round(avg_response_time, 2),
            "median_response_time": self._get_median(self.metrics["response_times"]),
            "complexity_distribution": self.metrics["complexity_distribution"],
            "domain_usage": self.metrics["domain_usage"],
            "worker_contributions": self.metrics["worker_contributions"],
            "validation_stats": self.metrics["validation_stats"],
            "recent_queries": self.metrics["query_history"][-10:]  # Last 10 queries
        }
        
        return stats
    
    def _get_median(self, values):
        """Calculate median of a list."""
        if not values:
            return 0
        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2
        if n % 2 == 0:
            return round((sorted_values[mid-1] + sorted_values[mid]) / 2, 2)
        return round(sorted_values[mid], 2)
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from JSON file if it exists."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.metrics = json.load(f)
                print(f"✅ Loaded existing metrics from {self.save_path}")
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.__init__(self.save_path)
        self.save_metrics()