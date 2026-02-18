"""Tests for CurriculumTrainer and BootstrapManager.

All HTTP calls and subprocess calls are mocked so tests are fast and offline.
"""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tensor.integrated_hdv import IntegratedHDVSystem


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def hdv(tmp_path):
    return IntegratedHDVSystem(
        hdv_dim=500, n_modes=5, embed_dim=32,
        library_path=str(tmp_path / "lib.json"),
    )


@pytest.fixture
def navigator(tmp_path):
    """A DeepWikiNavigator with rate limit 0 and tmp cache dir."""
    from tensor.deepwiki_navigator import DeepWikiNavigator
    return DeepWikiNavigator(
        cache_dir=str(tmp_path / "cache"),
        rate_limit_seconds=0,
    )


# ── CurriculumTrainer ─────────────────────────────────────────────────────────


class TestCurriculumTrainer:
    def test_init_creates_trainer(self, hdv, navigator, tmp_path):
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv,
            navigator=navigator,
            save_path=str(tmp_path / "patterns.json"),
        )
        assert trainer.pattern_count == 0

    def test_train_freecodecamp_empty(self, hdv, navigator, tmp_path):
        """Returns 0 when challenge extractor finds nothing."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        with patch.object(trainer.challenge_extractor, "extract_challenges",
                          return_value=[]):
            n = trainer.train_freecodecamp()
        assert n == 0

    def test_train_freecodecamp_encodes_challenges(self, hdv, navigator, tmp_path):
        """Challenges are encoded into HDV and recorded in discovery."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        fake_challenges = [
            {"id": "rev-str", "title": "Reverse String",
             "difficulty": "basic", "solution_pattern": "string_manipulation"},
            {"id": "fib",     "title": "Fibonacci",
             "difficulty": "intermediate", "solution_pattern": "recursion"},
        ]
        with patch.object(trainer.challenge_extractor, "extract_challenges",
                          return_value=fake_challenges):
            n = trainer.train_freecodecamp()

        assert n == 2
        assert trainer.pattern_count == 2
        assert trainer.discovery.pattern_count("execution") == 2

    def test_train_books_empty(self, hdv, navigator, tmp_path):
        """Returns 0 when book extractor finds nothing."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        with patch.object(trainer.book_extractor, "extract_book_curriculum",
                          return_value=[]):
            n = trainer.train_books()
        assert n == 0

    def test_train_books_encodes_books(self, hdv, navigator, tmp_path):
        """Books are encoded into math dimension."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        fake_books = [
            {"title": "Learn Python", "topic": "python", "format": "PDF"},
            {"title": "JS Guide",     "topic": "javascript", "format": "HTML"},
        ]
        with patch.object(trainer.book_extractor, "extract_book_curriculum",
                          return_value=fake_books):
            n = trainer.train_books()

        assert n == 2
        assert trainer.discovery.pattern_count("math") == 2

    def test_train_geometry_encodes_per_repo(self, hdv, navigator, tmp_path):
        """Each geometry repo gets encoded into physical dimension."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        fake_data = {
            "repo_url": "isl-org/Open3D",
            "summary":  "Open3D is a 3D data processing library",
            "file_tree": [],
            "key_files": ["setup.py"],
            "dependencies": [],
            "insights": {"patterns": ["src", "examples"], "language": "C++",
                         "complexity": "high", "star_count": 9000},
        }
        with patch.object(navigator, "navigate_repo", return_value=fake_data):
            n = trainer.train_geometry()

        # CURRICULUM["geometry"] has 2 repos (Open3D + PrusaSlicer)
        assert n == 2
        assert trainer.discovery.pattern_count("physical") == 2

    def test_train_architecture_encodes_behavioral(self, hdv, navigator, tmp_path):
        """Architecture repos go into behavioral dimension."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        fake_data = {
            "summary": "Node.js event-driven framework",
            "file_tree": [], "key_files": [], "dependencies": [],
            "insights": {"patterns": ["lib", "src"], "language": "JavaScript",
                         "complexity": "high", "star_count": 50000},
        }
        with patch.object(navigator, "navigate_repo", return_value=fake_data):
            n = trainer.train_architecture()

        assert n == 2   # 2 repos in CURRICULUM["architecture"]
        assert trainer.discovery.pattern_count("behavioral") == 2

    def test_train_calls_all_subtasks(self, hdv, navigator, tmp_path):
        """train() calls all four subtask methods."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        with patch.object(trainer, "train_freecodecamp", return_value=1) as m1, \
             patch.object(trainer, "train_books",         return_value=2) as m2, \
             patch.object(trainer, "train_geometry",      return_value=3) as m3, \
             patch.object(trainer, "train_architecture",  return_value=4) as m4:
            total = trainer.train()

        assert total == 10
        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_called_once()
        m4.assert_called_once()

    def test_save_and_reload_patterns(self, hdv, navigator, tmp_path):
        """Patterns saved to JSON are reloaded on next init."""
        from tensor.curriculum_trainer import CurriculumTrainer
        save_path = str(tmp_path / "p.json")
        trainer = CurriculumTrainer(hdv_system=hdv, navigator=navigator,
                                    save_path=save_path)
        fake_challenges = [
            {"id": "a", "title": "A", "difficulty": "basic",
             "solution_pattern": "general"},
        ]
        with patch.object(trainer.challenge_extractor, "extract_challenges",
                          return_value=fake_challenges):
            trainer.train_freecodecamp()
        trainer.save_patterns()
        assert trainer.pattern_count == 1

        trainer2 = CurriculumTrainer(hdv_system=hdv, navigator=navigator,
                                     save_path=save_path)
        assert trainer2.pattern_count == 1

    def test_discover_new_capabilities_encodes_relevant_repos(
        self, hdv, navigator, tmp_path
    ):
        """discover_new_capabilities encodes repos with relevance > 0."""
        from tensor.curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(
            hdv_system=hdv, navigator=navigator,
            save_path=str(tmp_path / "p.json"),
        )
        fake_gaps = [
            {
                "gap":  "gcode_generation",
                "repos": [
                    {"owner": "prusa3d", "name": "PrusaSlicer",
                     "relevance": 0.8,
                     "summary": "gcode slicer toolpath"},
                ],
            }
        ]
        with patch.object(trainer.capability_discovery,
                          "discover_repos_for_gaps", return_value=fake_gaps):
            gaps = trainer.discover_new_capabilities()

        assert len(gaps) == 1
        # The repo has relevance 0.8 → should have been encoded into physical dim
        assert trainer.discovery.pattern_count("physical") >= 1


# ── BootstrapManager ─────────────────────────────────────────────────────────


class TestBootstrapManager:
    @pytest.fixture
    def manager(self, hdv, tmp_path, monkeypatch):
        # Redirect capability_maps.json writes to tmp_path
        from tensor.bootstrap_manager import BootstrapManager
        mgr = BootstrapManager(hdv_system=hdv, rate_limit_seconds=0)
        return mgr

    # ── Scrapling ────────────────────────────────────────────────────────────

    def test_scrapling_install_failure_records_blocker(self, manager):
        with patch.object(manager, "_check_module", return_value=False), \
             patch.object(manager, "_pip_install",  return_value=False):
            success = manager.attempt_scrapling_integration()

        assert success is False
        assert any(b["resource"] == "Scrapling" for b in manager.blockers)

    def test_scrapling_already_installed_tries_fetch(self, manager, hdv):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = MagicMock()

        mock_scrapling = MagicMock()
        mock_scrapling.Fetcher.return_value = mock_fetcher

        with patch.object(manager, "_check_module", return_value=True), \
             patch("importlib.import_module", return_value=mock_scrapling):
            success = manager.attempt_scrapling_integration()

        assert success is True
        assert any("Scrapling" in msg for msg in manager.success_log)

    def test_scrapling_fetch_exception_records_blocker(self, manager):
        mock_scrapling = MagicMock()
        mock_scrapling.Fetcher.side_effect = RuntimeError("no browser")

        with patch.object(manager, "_check_module", return_value=True), \
             patch("importlib.import_module", return_value=mock_scrapling):
            success = manager.attempt_scrapling_integration()

        assert success is False
        assert any(b["resource"] == "Scrapling" for b in manager.blockers)

    # ── Open3D ───────────────────────────────────────────────────────────────

    def test_open3d_api_failure_uses_known_patterns(self, manager):
        """Falls back to known patterns when GitHub API is unreachable."""
        with patch.object(manager.session, "get",
                          return_value=MagicMock(status_code=403)):
            success = manager.attempt_open3d_integration()

        assert success is True
        assert any("Open3D" in msg for msg in manager.success_log)

    def test_open3d_encodes_known_operations(self, manager, hdv):
        """_encode_open3d_known encodes 5 workflows to physical HDV."""
        domain_dims_before = hdv.domain_masks.get("physical", None)
        manager._encode_open3d_known()
        # After encoding, physical domain should be registered
        assert "physical" in hdv.domain_masks

    def test_open3d_extract_geometry_patterns_rotation(self, manager):
        code = "mesh.rotate(R, center=[0,0,0])\n"
        patterns = manager._extract_geometry_patterns(code)
        names = [p["name"] for p in patterns]
        assert "rotation" in names

    def test_open3d_extract_geometry_patterns_translation(self, manager):
        code = "pcd.translate([1, 0, 0])\n"
        patterns = manager._extract_geometry_patterns(code)
        names = [p["name"] for p in patterns]
        assert "translation" in names

    def test_open3d_extract_empty_code(self, manager):
        patterns = manager._extract_geometry_patterns("")
        assert isinstance(patterns, list)

    # ── PrusaSlicer ──────────────────────────────────────────────────────────

    def test_prusaslicer_api_failure_uses_known(self, manager):
        with patch.object(manager.session, "get",
                          return_value=MagicMock(status_code=503)):
            success = manager.attempt_prusaslicer_integration()

        assert success is True
        assert any("PrusaSlicer" in msg for msg in manager.success_log)

    def test_prusaslicer_encode_known_produces_physical_patterns(self, manager, hdv):
        manager._encode_prusaslicer_known()
        assert "physical"  in hdv.domain_masks
        assert "behavioral" in hdv.domain_masks

    def test_prusaslicer_slicer_patterns_layer_height(self, manager, hdv):
        code = "float layer_height = 0.2f;\n"
        manager._encode_slicer_patterns_from_source(code)
        # Should not raise
        assert "physical" in hdv.domain_masks

    # ── Secret Knowledge ─────────────────────────────────────────────────────

    def test_secret_knowledge_404_records_blocker(self, manager):
        with patch.object(manager.session, "get",
                          return_value=MagicMock(status_code=404)):
            success = manager.attempt_secret_knowledge_integration()

        assert success is False
        assert any(b["resource"] == "SecretKnowledge" for b in manager.blockers)

    def test_secret_knowledge_parses_links(self, manager):
        readme = (
            "# Title\n"
            "- [Nmap](https://nmap.org) — network scanner\n"
            "- [Docker Docs](https://docs.docker.com) — container runtime\n"
        )
        with patch.object(manager.session, "get",
                          return_value=MagicMock(status_code=200, text=readme)):
            success = manager.attempt_secret_knowledge_integration()

        assert success is True
        assert any("SecretKnowledge" in msg for msg in manager.success_log)

    def test_categorise_links_networking(self, manager):
        links = [("Nmap Tool", "https://nmap.org"), ("Docker", "https://docker.com")]
        cats  = manager._categorise_links(links)
        assert "networking" in cats  # nmap → networking keyword

    def test_categorise_links_misc_fallback(self, manager):
        # Use a title + URL with no keyword matches (avoid "http" inside "https")
        links = [("Xyzzy Widget", "ftp://internal.corp/widget")]
        cats  = manager._categorise_links(links)
        assert "misc" in cats

    def test_categorise_links_empty(self, manager):
        cats = manager._categorise_links([])
        assert isinstance(cats, dict)

    # ── run_bootstrap ────────────────────────────────────────────────────────

    def test_run_bootstrap_returns_dict(self, manager):
        with patch.object(manager, "attempt_scrapling_integration",      return_value=True), \
             patch.object(manager, "attempt_open3d_integration",         return_value=True), \
             patch.object(manager, "attempt_prusaslicer_integration",    return_value=True), \
             patch.object(manager, "attempt_secret_knowledge_integration", return_value=True):
            results = manager.run_bootstrap()

        assert isinstance(results, dict)
        assert set(results.keys()) == {
            "Scrapling", "Open3D", "PrusaSlicer", "SecretKnowledge"
        }
        assert all(v is True for v in results.values())

    def test_run_bootstrap_partial_success(self, manager):
        with patch.object(manager, "attempt_scrapling_integration",      return_value=False), \
             patch.object(manager, "attempt_open3d_integration",         return_value=True), \
             patch.object(manager, "attempt_prusaslicer_integration",    return_value=True), \
             patch.object(manager, "attempt_secret_knowledge_integration", return_value=False):
            results = manager.run_bootstrap()

        assert results["Scrapling"]      is False
        assert results["Open3D"]         is True
        assert results["SecretKnowledge"] is False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def test_check_module_existing(self, manager):
        assert manager._check_module("json") is True

    def test_check_module_missing(self, manager):
        assert manager._check_module("nonexistent_package_xyz") is False

    def test_pip_install_calls_subprocess(self, manager):
        """_pip_install calls subprocess.run (not actually installing)."""
        with patch("subprocess.run") as mock_run, \
             patch.object(manager, "_check_module", return_value=True):
            mock_run.return_value = MagicMock(returncode=0)
            result = manager._pip_install("some_package")
        mock_run.assert_called_once()
        assert result is True

    def test_fetch_raw_github_404(self, manager):
        with patch.object(manager.session, "get",
                          return_value=MagicMock(status_code=404)):
            text = manager._fetch_raw_github("owner", "repo", "path/file.py")
        assert text == ""

    def test_fetch_raw_github_success(self, manager):
        with patch.object(manager.session, "get",
                          return_value=MagicMock(status_code=200, text="# code")):
            text = manager._fetch_raw_github("owner", "repo", "path/file.py")
        assert text == "# code"
