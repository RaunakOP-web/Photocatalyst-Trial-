"""
cache.py
SHA256-based caching system for incremental extraction and graph rebuilds.
"""

import os
import json
import hashlib

CACHE_FILE = ".graphify_cache/cache.json"

def calculate_sha256(filepath):
    """Calculates the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def load_cache():
    """Loads cache file, creating directories if necessary."""
    if not os.path.exists(CACHE_FILE):
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return {}

def save_cache(cache_data):
    """Saves the cache data to disk."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Failed to save cache: {e}")

class GraphifyCache:
    def __init__(self):
        self.cache = load_cache()
        self.new_cache = {}

    def is_changed(self, filepath):
        """
        Computes the file hash and checks if it has changed compared to the cache.
        Also records the new hash in the active cache.
        """
        current_hash = calculate_sha256(filepath)
        if not current_hash:
            return True
        
        self.new_cache[filepath] = current_hash
        
        # If file is not in cache, or hash differs, it has changed
        cached_hash = self.cache.get(filepath)
        return cached_hash != current_hash

    def commit(self):
        """Commits the active cache to disk."""
        save_cache(self.new_cache)
