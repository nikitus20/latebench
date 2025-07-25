"""
Enhanced Critic Result Storage System with Versioning and Caching

This module provides a comprehensive storage system for critic evaluation results
with features like versioning, intelligent caching, backup management, and
efficient query capabilities.
"""

import json
import sqlite3
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import threading
from contextlib import contextmanager

from src.critic import CriticResult
from src.metrics.deltabench import DeltaBenchMetrics

logger = logging.getLogger(__name__)


@dataclass
class StoredResult:
    """Stored critic result with metadata"""
    example_id: str
    critic_result: CriticResult
    model_version: str
    dataset_name: str
    evaluation_timestamp: str
    storage_version: str = "1.0"
    hash_signature: str = ""
    
    def __post_init__(self):
        if not self.hash_signature:
            self.hash_signature = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash signature for result integrity"""
        content = f"{self.example_id}:{self.model_version}:{self.evaluation_timestamp}"
        if self.critic_result:
            content += f":{self.critic_result.raw_response}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass 
class BatchResultSummary:
    """Summary of batch evaluation results"""
    batch_id: str
    dataset_name: str
    model_version: str
    total_examples: int
    successful_evaluations: int
    failed_evaluations: int
    cached_results: int
    evaluation_start: str
    evaluation_end: str
    deltabench_metrics: Optional[DeltaBenchMetrics]
    storage_path: str


class CriticResultStore:
    """Enhanced storage system for critic evaluation results"""
    
    def __init__(self, 
                 storage_dir: str = "./data/critic_store",
                 enable_sqlite: bool = True,
                 backup_retention_days: int = 30):
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.json_dir = self.storage_dir / "json"
        self.json_dir.mkdir(exist_ok=True)
        
        self.backup_dir = self.storage_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.cache_dir = self.storage_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # SQLite database for efficient querying
        self.enable_sqlite = enable_sqlite
        self.db_path = self.storage_dir / "critic_results.db"
        
        # Configuration
        self.backup_retention_days = backup_retention_days
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize storage
        self._initialize_storage()
        
        # In-memory cache
        self._memory_cache: Dict[str, StoredResult] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=1)
    
    def _initialize_storage(self):
        """Initialize storage systems"""
        if self.enable_sqlite:
            self._initialize_database()
        
        # Load recent results into memory cache
        self._load_recent_cache()
    
    def _initialize_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS critic_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    example_id TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    evaluation_timestamp TEXT NOT NULL,
                    storage_version TEXT NOT NULL,
                    hash_signature TEXT NOT NULL,
                    has_errors BOOLEAN NOT NULL,
                    error_steps TEXT,  -- JSON array
                    processing_time REAL,
                    result_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(example_id, model_version, dataset_name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT UNIQUE NOT NULL,
                    dataset_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    total_examples INTEGER NOT NULL,
                    successful_evaluations INTEGER NOT NULL,
                    failed_evaluations INTEGER NOT NULL,
                    cached_results INTEGER NOT NULL,
                    evaluation_start TEXT NOT NULL,
                    evaluation_end TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    deltabench_metrics TEXT,  -- JSON
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indices for efficient queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_example_model ON critic_results(example_id, model_version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON critic_results(dataset_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON critic_results(evaluation_timestamp)")
            
            conn.commit()
    
    def _load_recent_cache(self):
        """Load recent results into memory cache"""
        try:
            if not self.enable_sqlite:
                return
            
            # Load results from last 24 hours
            cutoff_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT example_id, result_path FROM critic_results 
                    WHERE evaluation_timestamp > ? 
                    ORDER BY evaluation_timestamp DESC 
                    LIMIT 1000
                """, (cutoff_time,))
                
                for row in cursor:
                    try:
                        result = self._load_json_result(row['result_path'])
                        if result:
                            cache_key = f"{row['example_id']}"
                            self._memory_cache[cache_key] = result
                            self._cache_expiry[cache_key] = datetime.utcnow() + self._cache_ttl
                    except Exception as e:
                        # Ensure _memory_cache and other attributes exist even if there's an error
                        if not hasattr(self, '_memory_cache'):
                            self._memory_cache = {}
                            self._cache_expiry = {}
                            self._cache_ttl = timedelta(hours=1)
                        logger.warning(f"Error loading cached result: {e}")
                        
        except Exception as e:
            # Ensure _memory_cache and other attributes exist even if there's an error
            if not hasattr(self, '_memory_cache'):
                self._memory_cache = {}
                self._cache_expiry = {}
                self._cache_ttl = timedelta(hours=1)
            logger.warning(f"Error loading cache: {e}")
    
    def store_result(self, 
                    example_id: str,
                    critic_result: CriticResult,
                    model_version: str,
                    dataset_name: str) -> str:
        """Store a single critic result"""
        
        with self._lock:
            # Create stored result
            stored_result = StoredResult(
                example_id=example_id,
                critic_result=critic_result,
                model_version=model_version,
                dataset_name=dataset_name,
                evaluation_timestamp=datetime.utcnow().isoformat()
            )
            
            # Generate storage path
            date_prefix = stored_result.evaluation_timestamp[:10]  # YYYY-MM-DD
            result_filename = f"{example_id}_{stored_result.hash_signature}.json"
            result_path = self.json_dir / date_prefix / result_filename
            result_path.parent.mkdir(exist_ok=True)
            
            # Save to JSON
            self._save_json_result(stored_result, str(result_path))
            
            # Update database
            if self.enable_sqlite:
                self._store_in_database(stored_result, str(result_path))
            
            # Update memory cache
            cache_key = f"{example_id}"
            self._memory_cache[cache_key] = stored_result
            self._cache_expiry[cache_key] = datetime.utcnow() + self._cache_ttl
            
            logger.debug(f"Stored result for {example_id} at {result_path}")
            return str(result_path)
    
    def store_batch_results(self,
                          results: Dict[str, CriticResult],
                          model_version: str,
                          dataset_name: str,
                          deltabench_metrics: Optional[DeltaBenchMetrics] = None) -> BatchResultSummary:
        """Store batch evaluation results"""
        
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow().isoformat()
        
        successful = 0
        failed = 0
        stored_paths = []
        
        # Store individual results
        for example_id, critic_result in results.items():
            try:
                if critic_result:
                    path = self.store_result(example_id, critic_result, model_version, dataset_name)
                    stored_paths.append(path)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error storing result for {example_id}: {e}")
                failed += 1
        
        # Create batch summary
        summary = BatchResultSummary(
            batch_id=batch_id,
            dataset_name=dataset_name,
            model_version=model_version,
            total_examples=len(results),
            successful_evaluations=successful,
            failed_evaluations=failed,
            cached_results=0,  # Would need to track this from evaluation
            evaluation_start=start_time,
            evaluation_end=datetime.utcnow().isoformat(),
            deltabench_metrics=deltabench_metrics,
            storage_path=str(self.json_dir)
        )
        
        # Store batch summary
        self._store_batch_summary(summary)
        
        logger.info(f"Stored batch {batch_id}: {successful} successful, {failed} failed")
        return summary
    
    def get_result(self, 
                  example_id: str, 
                  model_version: Optional[str] = None,
                  dataset_name: Optional[str] = None) -> Optional[StoredResult]:
        """Retrieve a stored result"""
        
        cache_key = f"{example_id}"
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            if self._cache_expiry.get(cache_key, datetime.min) > datetime.utcnow():
                return self._memory_cache[cache_key]
            else:
                # Cache expired
                del self._memory_cache[cache_key]
                if cache_key in self._cache_expiry:
                    del self._cache_expiry[cache_key]
        
        # Query database
        if self.enable_sqlite:
            return self._get_from_database(example_id, model_version, dataset_name)
        
        return None
    
    def get_batch_results(self,
                         dataset_name: str,
                         model_version: Optional[str] = None,
                         limit: Optional[int] = None) -> Dict[str, StoredResult]:
        """Get all results for a dataset"""
        
        if not self.enable_sqlite:
            return {}
        
        results = {}
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT example_id, result_path FROM critic_results WHERE dataset_name = ?"
            params = [dataset_name]
            
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            
            query += " ORDER BY evaluation_timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor = conn.execute(query, params)
            
            for row in cursor:
                try:
                    result = self._load_json_result(row['result_path'])
                    if result:
                        results[row['example_id']] = result
                except Exception as e:
                    logger.warning(f"Error loading result: {e}")
        
        return results
    
    def list_datasets(self) -> List[Tuple[str, str, int]]:  # (dataset_name, model_version, count)
        """List all available datasets with result counts"""
        
        if not self.enable_sqlite:
            return []
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT dataset_name, model_version, COUNT(*) as count
                FROM critic_results 
                GROUP BY dataset_name, model_version
                ORDER BY dataset_name, model_version
            """)
            
            return [(row[0], row[1], row[2]) for row in cursor]
    
    def get_batch_summaries(self, 
                           dataset_name: Optional[str] = None,
                           limit: int = 50) -> List[BatchResultSummary]:
        """Get batch evaluation summaries"""
        
        if not self.enable_sqlite:
            return []
        
        summaries = []
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM batch_summaries"
            params = []
            
            if dataset_name:
                query += " WHERE dataset_name = ?"
                params.append(dataset_name)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            for row in cursor:
                deltabench_metrics = None
                if row['deltabench_metrics']:
                    try:
                        metrics_data = json.loads(row['deltabench_metrics'])
                        deltabench_metrics = DeltaBenchMetrics(**metrics_data)
                    except Exception as e:
                        logger.warning(f"Error loading DeltaBench metrics: {e}")
                
                summary = BatchResultSummary(
                    batch_id=row['batch_id'],
                    dataset_name=row['dataset_name'],
                    model_version=row['model_version'],
                    total_examples=row['total_examples'],
                    successful_evaluations=row['successful_evaluations'],
                    failed_evaluations=row['failed_evaluations'],
                    cached_results=row['cached_results'],
                    evaluation_start=row['evaluation_start'],
                    evaluation_end=row['evaluation_end'],
                    deltabench_metrics=deltabench_metrics,
                    storage_path=row['storage_path']
                )
                
                summaries.append(summary)
        
        return summaries
    
    def create_backup(self) -> str:
        """Create backup of all stored results"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"critic_results_backup_{timestamp}"
        
        # Copy JSON files
        if self.json_dir.exists():
            shutil.copytree(self.json_dir, backup_path / "json")
        
        # Copy database
        if self.db_path.exists():
            shutil.copy2(self.db_path, backup_path / "critic_results.db")
        
        # Create backup manifest
        manifest = {
            'backup_timestamp': timestamp,
            'json_files_count': len(list(self.json_dir.rglob("*.json"))),
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0,
            'storage_version': '1.0'
        }
        
        with open(backup_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created backup at {backup_path}")
        return str(backup_path)
    
    def cleanup_old_backups(self):
        """Remove old backups beyond retention period"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.backup_retention_days)
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_dir.name.startswith("critic_results_backup_"):
                try:
                    # Extract timestamp from directory name
                    timestamp_str = backup_dir.name.replace("critic_results_backup_", "")
                    backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if backup_date < cutoff_date:
                        shutil.rmtree(backup_dir)
                        logger.info(f"Removed old backup: {backup_dir}")
                        
                except Exception as e:
                    logger.warning(f"Error processing backup {backup_dir}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage system statistics"""
        
        stats = {
            'storage_dir': str(self.storage_dir),
            'json_files_count': len(list(self.json_dir.rglob("*.json"))),
            'cache_size': len(self._memory_cache),
            'database_enabled': self.enable_sqlite
        }
        
        if self.enable_sqlite and self.db_path.exists():
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM critic_results")
                stats['database_records'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM batch_summaries")
                stats['batch_summaries'] = cursor.fetchone()[0]
                
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
        
        return stats
    
    # Private methods
    
    def _save_json_result(self, stored_result: StoredResult, path: str):
        """Save result to JSON file"""
        data = {
            'example_id': stored_result.example_id,
            'model_version': stored_result.model_version,
            'dataset_name': stored_result.dataset_name,
            'evaluation_timestamp': stored_result.evaluation_timestamp,
            'storage_version': stored_result.storage_version,
            'hash_signature': stored_result.hash_signature,
            'critic_result': stored_result.critic_result.to_dict()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_json_result(self, path: str) -> Optional[StoredResult]:
        """Load result from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            critic_result = CriticResult(**data['critic_result'])
            
            return StoredResult(
                example_id=data['example_id'],
                critic_result=critic_result,
                model_version=data['model_version'],
                dataset_name=data['dataset_name'],
                evaluation_timestamp=data['evaluation_timestamp'],
                storage_version=data.get('storage_version', '1.0'),
                hash_signature=data['hash_signature']
            )
            
        except Exception as e:
            logger.error(f"Error loading JSON result from {path}: {e}")
            return None
    
    def _store_in_database(self, stored_result: StoredResult, result_path: str):
        """Store result metadata in database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO critic_results 
                (example_id, dataset_name, model_version, evaluation_timestamp, 
                 storage_version, hash_signature, has_errors, error_steps, 
                 processing_time, result_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stored_result.example_id,
                stored_result.dataset_name,
                stored_result.model_version,
                stored_result.evaluation_timestamp,
                stored_result.storage_version,
                stored_result.hash_signature,
                stored_result.critic_result.has_errors,
                json.dumps(stored_result.critic_result.error_steps),
                stored_result.critic_result.processing_time,
                result_path,
                datetime.utcnow().isoformat()
            ))
            conn.commit()
    
    def _get_from_database(self, 
                          example_id: str,
                          model_version: Optional[str] = None,
                          dataset_name: Optional[str] = None) -> Optional[StoredResult]:
        """Get result from database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT result_path FROM critic_results WHERE example_id = ?"
            params = [example_id]
            
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            
            if dataset_name:
                query += " AND dataset_name = ?"
                params.append(dataset_name)
            
            query += " ORDER BY evaluation_timestamp DESC LIMIT 1"
            
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return self._load_json_result(row['result_path'])
        
        return None
    
    def _store_batch_summary(self, summary: BatchResultSummary):
        """Store batch summary in database"""
        if not self.enable_sqlite:
            return
        
        deltabench_json = None
        if summary.deltabench_metrics:
            deltabench_json = json.dumps(summary.deltabench_metrics.to_dict())
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO batch_summaries
                (batch_id, dataset_name, model_version, total_examples,
                 successful_evaluations, failed_evaluations, cached_results,
                 evaluation_start, evaluation_end, storage_path,
                 deltabench_metrics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.batch_id,
                summary.dataset_name,
                summary.model_version,
                summary.total_examples,
                summary.successful_evaluations,
                summary.failed_evaluations,
                summary.cached_results,
                summary.evaluation_start,
                summary.evaluation_end,
                summary.storage_path,
                deltabench_json,
                datetime.utcnow().isoformat()
            ))
            conn.commit()


# Convenience functions and context managers

@contextmanager
def critic_store_context(storage_dir: str = "./data/critic_store"):
    """Context manager for critic result store"""
    store = CriticResultStore(storage_dir)
    try:
        yield store
    finally:
        # Cleanup if needed
        pass


def get_default_store() -> CriticResultStore:
    """Get default critic result store instance"""
    return CriticResultStore()


if __name__ == "__main__":
    # Example usage
    store = CriticResultStore()
    stats = store.get_storage_stats()
    print("Storage stats:", stats)
    
    datasets = store.list_datasets()
    print("Available datasets:", datasets)