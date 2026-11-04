"""Phase 0 Environment & Schema Sanity Tests.

Verifies repository integrity, configuration files, schema formatting, and
essential project paths.
Works both with pytest and direct python execution via standard library unittest.
"""

import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPhase0Environment(unittest.TestCase):
    """Sanity checks for repository structure and schema contracts."""

    def test_required_directories_exist(self):
        """Ensure all core architectural directories are present."""
        required_dirs = [
            "configs",
            "dataset/raw",
            "dataset/processed",
            "research",
            "ml_engine",
            "traffic_monitor",
            "backend",
            "results",
            "documentation",
            "scripts",
        ]
        for d in required_dirs:
            dir_path = REPO_ROOT / d
            self.assertTrue(dir_path.exists(), f"Missing required directory: {d}")
            self.assertTrue(dir_path.is_dir(), f"Path is not a directory: {d}")

    def test_feature_schema_validity(self):
        """Verify feature_schema.json structure and completeness."""
        schema_path = REPO_ROOT / "ml_engine" / "feature_schema.json"
        self.assertTrue(schema_path.exists(), "feature_schema.json not found in ml_engine/")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        self.assertIn("schema_version", schema)
        self.assertIn("features", schema)
        self.assertIn("labels", schema)
        self.assertIsInstance(schema["features"], list)
        self.assertGreaterEqual(len(schema["features"]), 20, "Feature list should contain >= 20 features")

        for feature in schema["features"]:
            self.assertIn("name", feature, f"Feature missing name: {feature}")
            self.assertIn("type", feature, f"Feature {feature.get('name')} missing type")
            self.assertIn("unit", feature, f"Feature {feature.get('name')} missing unit")

        self.assertIn("BENIGN", schema["labels"], "BENIGN class must be present in labels")

    def test_yaml_configs_exist_and_nonempty(self):
        """Ensure all YAML configuration files exist and are populated."""
        config_files = [
            "configs/data_config.yaml",
            "configs/model_config.yaml",
            "configs/monitor_config.yaml",
        ]
        for cfg in config_files:
            cfg_path = REPO_ROOT / cfg
            self.assertTrue(cfg_path.exists(), f"Configuration file missing: {cfg}")
            content = cfg_path.read_text(encoding="utf-8").strip()
            self.assertGreater(len(content), 20, f"Config {cfg} appears unexpectedly empty")

    def test_env_example_present(self):
        """Ensure .env.example contains essential runtime variables."""
        env_path = REPO_ROOT / ".env.example"
        self.assertTrue(env_path.exists(), ".env.example not found")

        content = env_path.read_text(encoding="utf-8")
        expected_vars = [
            "SECRET_KEY",
            "DATABASE_URL",
            "FLOW_ACTIVE_TIMEOUT",
            "FLOW_IDLE_TIMEOUT",
            "SUPERVISED_THRESHOLD",
            "ANOMALY_THRESHOLD",
        ]
        for var in expected_vars:
            self.assertIn(var, content, f"Expected variable {var} in .env.example")


if __name__ == "__main__":
    unittest.main()
