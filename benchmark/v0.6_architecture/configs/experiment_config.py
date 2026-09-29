"""
Experiment configuration schema for ALLI v0.6 Architecture Benchmark.

This module defines the structured representation of a benchmark
experiment's configuration—the complete specification of what was
tested, on which candidate, with which dataset/scenarios, under which
environment/models, using which seeds.

Purpose:
    Enable complete reproducibility of benchmark runs.

Scope:
    Configuration describes an experiment's input parameters.
    Configuration does NOT contain results (those belong to result schema).

Design:
    All fields needed to reproduce an experiment are explicitly
    represented. The configuration is serializable and deserializable
    deterministically. Experiment identity is deterministic.

Semantics:

experiment_id (str):
    Unique identifier for one concrete benchmark execution/run.
    Generated as UUID; different for each execution even of same config.
    Identifies WHERE in time/space this run occurred.

configuration_fingerprint (str):
    Deterministic SHA256 hash of experimental parameters only.
    EXCLUDES: experiment_id, timestamp.
    INCLUDES: candidate, execution profile, versions, models, database,
    hardware, source revision, random seed.
    Same for logically equivalent configurations.
    Enables grouping results by experimental parameters.

This distinction allows:
- Run #1 and Run #2 same config = different experiment_id, same fingerprint
- Grouping results by "which experiments were identical in parameters"
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from types import MappingProxyType
import json
from uuid import uuid4
import hashlib


# Import versions from authoritative source
try:
    from .. import __version__ as version_module
except ImportError:
    # Direct import: load from the actual __version__.py file
    import importlib.util
    import os
    _version_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "__version__.py"
    )
    if os.path.exists(_version_file):
        spec = importlib.util.spec_from_file_location("v06_version", _version_file)
        version_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(version_module)
    else:
        class version_module:
            BENCHMARK_VERSION = "0.6.0"
            DATASET_VERSION = "0.1.0"
            SCENARIO_VERSION = "0.1.0"
            CANDIDATE_CONTRACT_VERSION = "0.1.0"
            RESULT_SCHEMA_VERSION = "0.1.0"


class CandidateType(Enum):
    """Candidate architecture enumeration."""
    CANDIDATE_A = "CANDIDATE_A"  # SQLite-centric
    CANDIDATE_B = "CANDIDATE_B"  # SQLite + Graph
    CANDIDATE_C = "CANDIDATE_C"  # Graph-native


class ExecutionProfile(Enum):
    """Execution profile for benchmark runner."""
    ISOLATED = "ISOLATED"          # Each scenario executes independently
    SEQUENTIAL_STATEFUL = "SEQUENTIAL_STATEFUL"  # Sequential with state persistence


@dataclass(frozen=True)
class CandidateConfiguration:
    """Candidate identity and version."""
    candidate_type: CandidateType
    candidate_version: str

    def __post_init__(self):
        if not self.candidate_version:
            raise ValueError("candidate_version cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_type": self.candidate_type.value,
            "candidate_version": self.candidate_version,
        }


@dataclass(frozen=True)
class VersionConfiguration:
    """Benchmark and schema versions."""
    benchmark_version: str
    dataset_version: str
    scenario_version: str
    contract_version: str
    result_schema_version: str

    def __post_init__(self):
        for name, value in [
            ("benchmark_version", self.benchmark_version),
            ("dataset_version", self.dataset_version),
            ("scenario_version", self.scenario_version),
            ("contract_version", self.contract_version),
            ("result_schema_version", self.result_schema_version),
        ]:
            if not value:
                raise ValueError(f"{name} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "dataset_version": self.dataset_version,
            "scenario_version": self.scenario_version,
            "contract_version": self.contract_version,
            "result_schema_version": self.result_schema_version,
        }


@dataclass(frozen=True)
class ModelConfiguration:
    """Model identity for experiment reproducibility."""
    primary_model: str
    embedding_model: Optional[str] = None
    reranker: Optional[str] = None

    def __post_init__(self):
        if not self.primary_model:
            raise ValueError("primary_model cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_model": self.primary_model,
            "embedding_model": self.embedding_model,
            "reranker": self.reranker,
        }


@dataclass(frozen=True)
class DatabaseConfiguration:
    """Database/runtime configuration (descriptive)."""
    database_engine: str  # e.g., "sqlite", "graph", "hybrid"
    engine_version: str
    configuration_profile: str = "default"
    storage_mode: Optional[str] = None  # e.g., "in-memory", "file-based"
    index_configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.database_engine:
            raise ValueError("database_engine cannot be empty")
        if not self.engine_version:
            raise ValueError("engine_version cannot be empty")

        # FIX 2: Make nested dict immutable
        object.__setattr__(
            self,
            'index_configuration',
            MappingProxyType(self.index_configuration)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_engine": self.database_engine,
            "engine_version": self.engine_version,
            "configuration_profile": self.configuration_profile,
            "storage_mode": self.storage_mode,
            "index_configuration": dict(self.index_configuration),  # Convert back from MappingProxyType
        }


@dataclass(frozen=True)
class HardwareConfiguration:
    """Hardware and environment metadata."""
    cpu_count: Optional[int] = None
    cpu_model: Optional[str] = None
    ram_gb: Optional[float] = None
    gpu_count: Optional[int] = None
    gpu_model: Optional[str] = None
    vram_gb: Optional[float] = None
    operating_system: str = "unknown"
    os_version: Optional[str] = None
    python_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "cpu_model": self.cpu_model,
            "ram_gb": self.ram_gb,
            "gpu_count": self.gpu_count,
            "gpu_model": self.gpu_model,
            "vram_gb": self.vram_gb,
            "operating_system": self.operating_system,
            "os_version": self.os_version,
            "python_version": self.python_version,
        }


@dataclass(frozen=True)
class SourceRevision:
    """Source code revision information."""
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    revision_source: str = "git"  # "git", "unknown", etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "revision_source": self.revision_source,
        }


@dataclass(frozen=True)
class ExperimentConfiguration:
    """
    Complete experiment configuration for benchmark reproducibility.

    This class represents a complete, immutable specification of a
    benchmark experiment. Every field is necessary to reproduce the
    experiment.

    Not present: results, metrics, failures, answers, evidence, or any
    benchmark output. Those belong to the result schema.

    Semantics:

    experiment_id: identifies this concrete execution/run
    timestamp: when this run executed (metadata, not part of config fingerprint)
    configuration_fingerprint: deterministic hash of experimental parameters
    """

    # Execution identity
    experiment_id: str
    timestamp: datetime

    # What was tested
    candidate: CandidateConfiguration
    execution_profile: ExecutionProfile

    # Schema versions
    versions: VersionConfiguration

    # Models
    models: ModelConfiguration

    # Database/runtime
    database: DatabaseConfiguration

    # Hardware/environment
    hardware: HardwareConfiguration

    # Source code
    source_revision: SourceRevision

    # Determinism
    random_seed: int

    def __post_init__(self):
        # Validate all required fields
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")

    def validate(self) -> None:
        """Validate that the configuration is complete and valid."""
        # Delegates to __post_init__ + nested dataclass validations
        pass

    def to_dict(self, include_timestamp: bool = True) -> dict[str, Any]:
        """
        Serialize configuration to dictionary.

        Args:
            include_timestamp: if True, includes timestamp in output.
                When False, timestamp is excluded (used for fingerprinting).
        """
        self.validate()
        data = {
            "experiment_id": self.experiment_id,
            "candidate": self.candidate.to_dict(),
            "execution_profile": self.execution_profile.value,
            "versions": self.versions.to_dict(),
            "models": self.models.to_dict(),
            "database": self.database.to_dict(),
            "hardware": self.hardware.to_dict(),
            "source_revision": self.source_revision.to_dict(),
            "random_seed": self.random_seed,
        }
        if include_timestamp:
            data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_json(self, include_timestamp: bool = True) -> str:
        """Serialize configuration to JSON string (deterministic)."""
        data = self.to_dict(include_timestamp=include_timestamp)
        # Use sort_keys to ensure deterministic JSON output
        return json.dumps(data, sort_keys=True, indent=2)

    @property
    def configuration_fingerprint(self) -> str:
        """
        FIX 1: Deterministic fingerprint of configuration parameters.

        Excludes:
        - experiment_id (run identity)
        - timestamp (execution metadata)

        Includes all experimental parameters that affect reproducibility:
        - candidate identity/version
        - execution profile
        - all version fields
        - model configuration
        - database configuration
        - hardware metadata
        - source revision
        - random seed

        Two configurations with identical parameters produce identical fingerprints.
        """
        # Get config without experiment_id or timestamp (both are run identity, not config parameters)
        config_data = {
            "candidate": self.candidate.to_dict(),
            "execution_profile": self.execution_profile.value,
            "versions": self.versions.to_dict(),
            "models": self.models.to_dict(),
            "database": self.database.to_dict(),
            "hardware": self.hardware.to_dict(),
            "source_revision": self.source_revision.to_dict(),
            "random_seed": self.random_seed,
        }
        json_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExperimentConfiguration":
        """Deserialize configuration from dictionary."""
        return ExperimentConfiguration(
            experiment_id=data["experiment_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            candidate=CandidateConfiguration(
                candidate_type=CandidateType(data["candidate"]["candidate_type"]),
                candidate_version=data["candidate"]["candidate_version"],
            ),
            execution_profile=ExecutionProfile(data["execution_profile"]),
            versions=VersionConfiguration(
                benchmark_version=data["versions"]["benchmark_version"],
                dataset_version=data["versions"]["dataset_version"],
                scenario_version=data["versions"]["scenario_version"],
                contract_version=data["versions"]["contract_version"],
                result_schema_version=data["versions"]["result_schema_version"],
            ),
            models=ModelConfiguration(
                primary_model=data["models"]["primary_model"],
                embedding_model=data["models"].get("embedding_model"),
                reranker=data["models"].get("reranker"),
            ),
            database=DatabaseConfiguration(
                database_engine=data["database"]["database_engine"],
                engine_version=data["database"]["engine_version"],
                configuration_profile=data["database"].get("configuration_profile", "default"),
                storage_mode=data["database"].get("storage_mode"),
                index_configuration=data["database"].get("index_configuration", {}),
            ),
            hardware=HardwareConfiguration(
                cpu_count=data["hardware"].get("cpu_count"),
                cpu_model=data["hardware"].get("cpu_model"),
                ram_gb=data["hardware"].get("ram_gb"),
                gpu_count=data["hardware"].get("gpu_count"),
                gpu_model=data["hardware"].get("gpu_model"),
                vram_gb=data["hardware"].get("vram_gb"),
                operating_system=data["hardware"].get("operating_system", "unknown"),
                os_version=data["hardware"].get("os_version"),
                python_version=data["hardware"].get("python_version", "unknown"),
            ),
            source_revision=SourceRevision(
                git_commit=data["source_revision"].get("git_commit"),
                git_branch=data["source_revision"].get("git_branch"),
                git_dirty=data["source_revision"].get("git_dirty", False),
                revision_source=data["source_revision"].get("revision_source", "git"),
            ),
            random_seed=data["random_seed"],
        )

    @staticmethod
    def from_json(json_str: str) -> "ExperimentConfiguration":
        """Deserialize configuration from JSON string."""
        data = json.loads(json_str)
        return ExperimentConfiguration.from_dict(data)


class ExperimentConfigurationBuilder:
    """
    FIX 3: Builder with explicit distinction between fixture and real config.

    This builder is a CONVENIENCE TOOL FOR TESTING.

    DO NOT use this builder for production experiments without explicit
    values for all critical fields.

    For production use, construct ExperimentConfiguration directly with
    explicit values.
    """

    def __init__(self, testing: bool = True):
        """
        Initialize builder.

        Args:
            testing: if True, uses test fixture defaults; if False,
                requires all critical fields to be explicitly set.
        """
        self.testing = testing
        self._candidate: Optional[CandidateConfiguration] = None
        self._execution_profile: Optional[ExecutionProfile] = None
        self._versions: Optional[VersionConfiguration] = None
        self._models: Optional[ModelConfiguration] = None
        self._database: Optional[DatabaseConfiguration] = None
        self._hardware: Optional[HardwareConfiguration] = None
        self._source_revision: Optional[SourceRevision] = None
        self._random_seed: Optional[int] = None

        if testing:
            # FIX 3: Test fixture defaults (visibly labeled as test values)
            self._candidate = CandidateConfiguration(
                candidate_type=CandidateType.CANDIDATE_A,
                candidate_version="0.1.0-test-fixture"
            )
            self._execution_profile = ExecutionProfile.ISOLATED
            # FIX 4: Import versions from authoritative source
            self._versions = VersionConfiguration(
                benchmark_version=version_module.BENCHMARK_VERSION,
                dataset_version=version_module.DATASET_VERSION,
                scenario_version=version_module.SCENARIO_VERSION,
                contract_version=version_module.CANDIDATE_CONTRACT_VERSION,
                result_schema_version=version_module.RESULT_SCHEMA_VERSION,
            )
            self._models = ModelConfiguration(primary_model="test-fixture-model")
            self._database = DatabaseConfiguration(
                database_engine="test-fixture",
                engine_version="0.1.0-test"
            )
            self._hardware = HardwareConfiguration(
                operating_system="test-fixture-os",
                python_version="3.11"
            )
            self._source_revision = SourceRevision(revision_source="test-fixture")
            self._random_seed = 42

    def with_candidate(self, candidate_type: CandidateType, version: str) -> "ExperimentConfigurationBuilder":
        """Set candidate identity."""
        self._candidate = CandidateConfiguration(candidate_type=candidate_type, candidate_version=version)
        return self

    def with_execution_profile(self, profile: ExecutionProfile) -> "ExperimentConfigurationBuilder":
        """Set execution profile."""
        self._execution_profile = profile
        return self

    def with_versions(self, benchmark: str, dataset: str, scenario: str,
                      contract: str, result_schema: str) -> "ExperimentConfigurationBuilder":
        """Set all versions explicitly."""
        self._versions = VersionConfiguration(
            benchmark_version=benchmark,
            dataset_version=dataset,
            scenario_version=scenario,
            contract_version=contract,
            result_schema_version=result_schema,
        )
        return self

    def with_models(self, primary: str, embedding: Optional[str] = None, reranker: Optional[str] = None) -> "ExperimentConfigurationBuilder":
        """Set model configuration."""
        self._models = ModelConfiguration(primary_model=primary, embedding_model=embedding, reranker=reranker)
        return self

    def with_database(self, engine: str, version: str, profile: str = "default") -> "ExperimentConfigurationBuilder":
        """Set database configuration."""
        self._database = DatabaseConfiguration(
            database_engine=engine, engine_version=version, configuration_profile=profile
        )
        return self

    def with_random_seed(self, seed: int) -> "ExperimentConfigurationBuilder":
        """Set random seed."""
        self._random_seed = seed
        return self

    def with_git_commit(self, commit: str, branch: Optional[str] = None, dirty: bool = False) -> "ExperimentConfigurationBuilder":
        """Set git revision information."""
        self._source_revision = SourceRevision(git_commit=commit, git_branch=branch, git_dirty=dirty)
        return self

    def build(self) -> ExperimentConfiguration:
        """
        Build the configuration.

        Raises ValueError if any critical field is missing/invalid.
        """
        if self._candidate is None:
            raise ValueError("candidate is required; use with_candidate()")
        if self._execution_profile is None:
            raise ValueError("execution_profile is required; use with_execution_profile()")
        if self._versions is None:
            raise ValueError("versions are required; use with_versions()")
        if self._models is None:
            raise ValueError("models are required; use with_models()")
        if self._database is None:
            raise ValueError("database is required; use with_database()")
        if self._random_seed is None:
            raise ValueError("random_seed is required; use with_random_seed()")

        config = ExperimentConfiguration(
            experiment_id=f"exp_{uuid4().hex[:12]}",
            timestamp=datetime.utcnow(),
            candidate=self._candidate,
            execution_profile=self._execution_profile,
            versions=self._versions,
            models=self._models,
            database=self._database,
            hardware=self._hardware or HardwareConfiguration(),
            source_revision=self._source_revision or SourceRevision(),
            random_seed=self._random_seed,
        )
        config.validate()
        return config

