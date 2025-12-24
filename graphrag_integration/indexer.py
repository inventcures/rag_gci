"""
GraphRAG Indexing Pipeline for Palli Sahayak

Provides document indexing capabilities using Microsoft GraphRAG.
Supports both standard (LLM-based) and fast (NLP-based) indexing methods.

Usage:
    from graphrag_integration.indexer import GraphRAGIndexer, IndexingMethod
    from graphrag_integration.config import GraphRAGConfig

    config = GraphRAGConfig.from_yaml("./data/graphrag/settings.yaml")
    indexer = GraphRAGIndexer(config)

    # Index all documents
    result = await indexer.index_documents()

    # Monitor progress
    print(f"Status: {indexer.status}, Progress: {indexer.progress}%")

    # Verify index
    verification = await indexer.verify_index()
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime

from graphrag_integration.config import GraphRAGConfig

logger = logging.getLogger(__name__)


class IndexingMethod(Enum):
    """
    Available indexing methods.

    STANDARD: LLM-based extraction (higher quality, slower, more expensive)
    FAST: NLP-based extraction (faster, cheaper, lower quality)
    """
    STANDARD = "standard"
    FAST = "fast"


class IndexingStatus(Enum):
    """
    Indexing job status.

    PENDING: Not yet started
    RUNNING: Currently indexing
    COMPLETED: Successfully finished
    FAILED: Indexing failed
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GraphRAGIndexer:
    """
    Document indexing pipeline for GraphRAG.

    Handles document processing, entity extraction, community detection,
    and embedding generation.

    Attributes:
        config: GraphRAG configuration
        method: Indexing method (standard or fast)
        status: Current indexing status
        progress: Progress percentage (0-100)

    Example:
        indexer = GraphRAGIndexer(config)

        # Add progress callback
        indexer.add_callback(lambda status, progress: print(f"{status}: {progress}%"))

        # Run indexing
        result = await indexer.index_documents()

        # Check results
        print(f"Entities: {result['entities_extracted']}")
        print(f"Communities: {result['communities_created']}")
    """

    def __init__(
        self,
        config: GraphRAGConfig,
        method: IndexingMethod = IndexingMethod.STANDARD
    ):
        """
        Initialize indexer.

        Args:
            config: GraphRAG configuration
            method: Indexing method to use
        """
        self.config = config
        self.method = method
        self.status = IndexingStatus.PENDING
        self.progress = 0
        self._callbacks: List[Callable[[str, int], None]] = []
        self._stats: Dict[str, Any] = {}
        self._error: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    async def index_documents(
        self,
        documents: Optional[List[str]] = None,
        update_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents using GraphRAG pipeline.

        Args:
            documents: Optional list of document paths (uses input_dir if None)
            update_mode: If True, perform incremental update instead of full reindex

        Returns:
            Dictionary with indexing results and statistics:
            - status: "success" or "error"
            - documents_processed: Number of documents
            - entities_extracted: Number of entities
            - relationships_extracted: Number of relationships
            - communities_created: Number of communities
            - text_units_created: Number of text chunks
            - duration_seconds: Time taken

        Raises:
            RuntimeError: If indexing fails
        """
        self.status = IndexingStatus.RUNNING
        self.progress = 0
        self._start_time = datetime.now()
        self._error = None

        try:
            self._notify_callbacks("Starting indexing pipeline...", 0)
            logger.info(f"Starting indexing with method: {self.method.value}")

            # Try to use native GraphRAG
            try:
                from graphrag.api import build_index

                native_config = self.config.to_graphrag_config()

                # Execute indexing pipeline
                if update_mode:
                    result = await self._run_update_index(native_config)
                else:
                    result = await self._run_full_index(native_config)

            except ImportError:
                logger.warning("GraphRAG not installed, using mock indexing")
                result = await self._mock_index()

            self.status = IndexingStatus.COMPLETED
            self.progress = 100
            self._end_time = datetime.now()

            # Calculate duration
            duration = (self._end_time - self._start_time).total_seconds()
            result["duration_seconds"] = duration
            result["status"] = "success"

            self._stats = result
            logger.info(f"Indexing completed in {duration:.1f}s")
            self._notify_callbacks("Indexing completed successfully", 100)

            return result

        except Exception as e:
            self.status = IndexingStatus.FAILED
            self._error = str(e)
            self._end_time = datetime.now()
            logger.error(f"Indexing failed: {e}")
            self._notify_callbacks(f"Indexing failed: {e}", self.progress)
            raise RuntimeError(f"Indexing failed: {e}") from e

    async def _run_full_index(self, config: Any) -> Dict[str, Any]:
        """
        Run full indexing pipeline using native GraphRAG.

        Args:
            config: Native GraphRAG configuration

        Returns:
            Indexing statistics
        """
        try:
            from graphrag.api.index import build_index

            self._notify_callbacks("Loading documents...", 10)

            # Build index
            await build_index(config=config)

            self._notify_callbacks("Collecting statistics...", 90)

            # Collect statistics
            return await self._collect_stats()

        except ImportError:
            logger.warning("GraphRAG API not available, using mock")
            return await self._mock_index()

    async def _run_update_index(self, config: Any) -> Dict[str, Any]:
        """
        Run incremental update indexing.

        Args:
            config: Native GraphRAG configuration

        Returns:
            Indexing statistics
        """
        try:
            from graphrag.api.index import build_index

            self._notify_callbacks("Running incremental update...", 10)

            # Build with update mode
            await build_index(config=config, update_index=True)

            self._notify_callbacks("Collecting statistics...", 90)

            return await self._collect_stats()

        except ImportError:
            logger.warning("GraphRAG API not available, using mock")
            return await self._mock_index()

    async def _mock_index(self) -> Dict[str, Any]:
        """
        Mock indexing for testing without GraphRAG installed.

        Returns:
            Mock statistics
        """
        steps = [
            ("Loading documents...", 10),
            ("Chunking text...", 20),
            ("Extracting entities...", 40),
            ("Extracting relationships...", 50),
            ("Building graph...", 60),
            ("Detecting communities...", 70),
            ("Generating community reports...", 80),
            ("Creating embeddings...", 90),
            ("Finalizing...", 95),
        ]

        for status, progress in steps:
            self.progress = progress
            self._notify_callbacks(status, progress)
            await asyncio.sleep(0.1)  # Simulate work

        # Return mock statistics
        return {
            "status": "mock",
            "documents_processed": self._count_input_documents(),
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "communities_created": 0,
            "text_units_created": 0,
            "mock": True,
        }

    def _count_input_documents(self) -> int:
        """Count documents in input directory."""
        input_dir = self.config.input_dir
        if not input_dir.exists():
            return 0

        count = 0
        for ext in ["*.txt", "*.md", "*.pdf", "*.docx"]:
            count += len(list(input_dir.glob(ext)))
        return count

    async def _collect_stats(self) -> Dict[str, Any]:
        """
        Collect indexing statistics from output files.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "method": self.method.value,
            "documents_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "communities_created": 0,
            "text_units_created": 0,
            "community_reports_created": 0,
        }

        try:
            import pandas as pd

            artifacts_dir = self.config.artifacts_dir

            # Count entities
            entities_path = artifacts_dir / "entities.parquet"
            if entities_path.exists():
                df = pd.read_parquet(entities_path)
                stats["entities_extracted"] = len(df)
                logger.info(f"Found {len(df)} entities")

            # Count relationships
            relationships_path = artifacts_dir / "relationships.parquet"
            if relationships_path.exists():
                df = pd.read_parquet(relationships_path)
                stats["relationships_extracted"] = len(df)
                logger.info(f"Found {len(df)} relationships")

            # Count communities
            communities_path = artifacts_dir / "communities.parquet"
            if communities_path.exists():
                df = pd.read_parquet(communities_path)
                stats["communities_created"] = len(df)
                logger.info(f"Found {len(df)} communities")

            # Count community reports
            reports_path = artifacts_dir / "community_reports.parquet"
            if reports_path.exists():
                df = pd.read_parquet(reports_path)
                stats["community_reports_created"] = len(df)
                logger.info(f"Found {len(df)} community reports")

            # Count text units
            text_units_path = artifacts_dir / "text_units.parquet"
            if text_units_path.exists():
                df = pd.read_parquet(text_units_path)
                stats["text_units_created"] = len(df)
                logger.info(f"Found {len(df)} text units")

            # Count documents
            documents_path = artifacts_dir / "documents.parquet"
            if documents_path.exists():
                df = pd.read_parquet(documents_path)
                stats["documents_processed"] = len(df)
                logger.info(f"Found {len(df)} documents")

        except ImportError:
            logger.warning("pandas not installed, cannot collect stats")
        except Exception as e:
            logger.warning(f"Failed to collect stats: {e}")

        return stats

    def add_callback(self, callback: Callable[[str, int], None]) -> None:
        """
        Add progress callback.

        Args:
            callback: Function(status: str, progress: int) -> None
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[str, int], None]) -> None:
        """
        Remove progress callback.

        Args:
            callback: Previously added callback function
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, status: str, progress: int) -> None:
        """
        Notify all callbacks of progress update.

        Args:
            status: Status message
            progress: Progress percentage (0-100)
        """
        self.progress = progress
        for callback in self._callbacks:
            try:
                callback(status, progress)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get indexing statistics.

        Returns:
            Dictionary with statistics from last indexing run
        """
        return self._stats.copy()

    def get_error(self) -> Optional[str]:
        """
        Get error message if indexing failed.

        Returns:
            Error message or None
        """
        return self._error

    def get_duration(self) -> Optional[float]:
        """
        Get indexing duration in seconds.

        Returns:
            Duration in seconds or None if not completed
        """
        if self._start_time and self._end_time:
            return (self._end_time - self._start_time).total_seconds()
        return None

    async def verify_index(self) -> Dict[str, Any]:
        """
        Verify index integrity.

        Returns:
            Dictionary with verification results:
            - valid: True if all checks pass
            - errors: List of error messages
            - files_checked: List of files verified
            - file_stats: Statistics for each file
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "files_checked": [],
            "file_stats": {},
        }

        required_files = [
            "entities.parquet",
            "relationships.parquet",
            "communities.parquet",
            "community_reports.parquet",
            "text_units.parquet",
        ]

        optional_files = [
            "documents.parquet",
            "embeddings.entity.parquet",
            "embeddings.text_unit.parquet",
        ]

        artifacts_dir = self.config.artifacts_dir

        # Check required files
        for filename in required_files:
            filepath = artifacts_dir / filename
            results["files_checked"].append(filename)

            if not filepath.exists():
                results["valid"] = False
                results["errors"].append(f"Missing required file: {filename}")
            else:
                # Try to read and validate the file
                try:
                    import pandas as pd
                    df = pd.read_parquet(filepath)
                    results["file_stats"][filename] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "size_bytes": filepath.stat().st_size,
                    }
                    if len(df) == 0:
                        results["warnings"].append(f"Empty file: {filename}")
                except ImportError:
                    results["warnings"].append("pandas not installed, cannot verify file contents")
                except Exception as e:
                    results["valid"] = False
                    results["errors"].append(f"Corrupted file {filename}: {e}")

        # Check optional files
        for filename in optional_files:
            filepath = artifacts_dir / filename
            if filepath.exists():
                results["files_checked"].append(filename)
                try:
                    import pandas as pd
                    df = pd.read_parquet(filepath)
                    results["file_stats"][filename] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "size_bytes": filepath.stat().st_size,
                    }
                except Exception as e:
                    results["warnings"].append(f"Could not read {filename}: {e}")

        return results

    async def clear_index(self) -> bool:
        """
        Clear all indexed data.

        Returns:
            True if successful
        """
        import shutil

        try:
            artifacts_dir = self.config.artifacts_dir
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir)
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleared index at {artifacts_dir}")

            # Clear cache
            cache_dir = self.config.cache_dir
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleared cache at {cache_dir}")

            self._stats = {}
            self.status = IndexingStatus.PENDING
            self.progress = 0

            return True

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    def __repr__(self) -> str:
        return (
            f"GraphRAGIndexer("
            f"method={self.method.value}, "
            f"status={self.status.value}, "
            f"progress={self.progress}%)"
        )

    def __str__(self) -> str:
        return self.__repr__()
