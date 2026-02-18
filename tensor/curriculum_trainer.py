"""
FICUTS Curriculum Trainer

Phase 2 of CLAUDE_CODE_IMPLEMENTATION_PLAN.

Uses DeepWikiNavigator to extract structured learning content from curated
repos and encodes each piece into HDV space via IntegratedHDVSystem, feeding
CrossDimensionalDiscovery for universal pattern detection.

Curriculum sources:
  - freeCodeCamp          → execution dimension (coding challenge patterns)
  - free-programming-books → math dimension (book/topic metadata)
  - Open3D / PrusaSlicer  → physical dimension (geometry/G-code workflows)
  - Node.js / Vue.js      → behavioral dimension (architectural patterns)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
from tensor.deepwiki_navigator import (
    CapabilityDiscovery,
    DeepWikiBookExtractor,
    DeepWikiChallengeExtractor,
    DeepWikiNavigator,
)


class CurriculumTrainer:
    """
    Train HDV system on structured curricula from curated repos.

    The trainer orchestrates four sub-extractors, encodes all found
    patterns into HDV space, and records them in CrossDimensionalDiscovery
    so universal detection can run across all dimensions.
    """

    CURRICULUM: Dict[str, List[tuple]] = {
        "challenges": [
            ("freeCodeCamp", "freeCodeCamp"),
        ],
        "books": [
            ("EbookFoundation", "free-programming-books"),
        ],
        "geometry": [
            ("isl-org",  "Open3D"),
            ("prusa3d",  "PrusaSlicer"),
        ],
        "architecture": [
            ("nodejs", "node"),
            ("vuejs",  "vue"),
        ],
    }

    def __init__(
        self,
        hdv_system,
        navigator: Optional[DeepWikiNavigator] = None,
        discovery: Optional[CrossDimensionalDiscovery] = None,
        save_path: str = "tensor/data/curriculum_patterns.json",
        rate_limit_seconds: float = 1.5,
    ):
        self.hdv_system = hdv_system
        self.navigator  = navigator or DeepWikiNavigator(
            rate_limit_seconds=rate_limit_seconds
        )
        self.discovery  = discovery or CrossDimensionalDiscovery(hdv_system)
        self.save_path  = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        rl = rate_limit_seconds
        self.challenge_extractor  = DeepWikiChallengeExtractor(self.navigator, rl)
        self.book_extractor       = DeepWikiBookExtractor(self.navigator, rl)
        self.capability_discovery = CapabilityDiscovery(self.navigator, hdv_system)

        self._patterns: List[Dict] = self._load_patterns()

    # ── Public API ──────────────────────────────────────────────────────────

    def train(self) -> int:
        """
        Run the full curriculum (challenges + books + geometry + architecture).

        Returns total patterns encoded.
        """
        n  = self.train_freecodecamp()
        n += self.train_books()
        n += self.train_geometry()
        n += self.train_architecture()
        self.save_patterns()
        return n

    def train_freecodecamp(
        self,
        owner: str = "freeCodeCamp",
        repo:  str = "freeCodeCamp",
    ) -> int:
        """Encode coding challenges into the execution dimension."""
        print(f"[Curriculum] Extracting challenges from {owner}/{repo}...")
        challenges = self.challenge_extractor.extract_challenges(owner, repo)
        print(f"[Curriculum]   Found {len(challenges)} challenges")

        encoded = 0
        for ch in challenges:
            workflow = [
                ch.get("solution_pattern", "general"),
                ch.get("difficulty", "basic"),
                ch["title"].lower(),
            ]
            hdv = self.hdv_system.encode_workflow(workflow, domain="execution")
            self.discovery.record_pattern(
                "execution", hdv,
                {
                    "type":    ch.get("solution_pattern", "general"),
                    "content": ch["title"],
                    "source":  "freecodecamp",
                },
            )
            self._patterns.append({"source": "freecodecamp", "id": ch["id"]})
            encoded += 1

        return encoded

    def train_books(
        self,
        owner: str = "EbookFoundation",
        repo:  str = "free-programming-books",
    ) -> int:
        """Encode book/topic metadata into the math dimension."""
        print(f"[Curriculum] Extracting books from {owner}/{repo}...")
        books = self.book_extractor.extract_book_curriculum(owner, repo)
        print(f"[Curriculum]   Found {len(books)} books")

        encoded = 0
        for book in books:
            workflow = [
                book.get("topic", "programming"),
                book["format"].lower(),
                "read",
            ]
            hdv = self.hdv_system.encode_workflow(workflow, domain="math")
            self.discovery.record_pattern(
                "math", hdv,
                {
                    "type":    book.get("topic", "programming"),
                    "content": book["title"],
                    "source":  "free-programming-books",
                },
            )
            self._patterns.append({"source": "books", "title": book["title"]})
            encoded += 1

        return encoded

    def train_geometry(self) -> int:
        """Encode geometry / G-code workflows into the physical dimension."""
        print("[Curriculum] Encoding geometry patterns (physical dimension)...")
        encoded = 0

        for owner, repo in self.CURRICULUM["geometry"]:
            print(f"[Curriculum]   Navigating {owner}/{repo}...")
            data = self.navigator.navigate_repo(owner, repo)
            if not data:
                continue

            workflow = [
                data["summary"][:50].lower(),
                *[Path(f).stem.lower() for f in data["key_files"][:3]],
                *data["insights"].get("patterns", [])[:3],
            ]
            hdv = self.hdv_system.encode_workflow(workflow, domain="physical")
            self.discovery.record_pattern(
                "physical", hdv,
                {
                    "type":    "geometry",
                    "content": f"{owner}/{repo}",
                    "source":  "github",
                },
            )
            self._patterns.append({"source": f"{owner}/{repo}", "type": "geometry"})
            encoded += 1

        return encoded

    def train_architecture(self) -> int:
        """Encode architectural patterns into the behavioral dimension."""
        print("[Curriculum] Encoding architecture patterns (behavioral dimension)...")
        encoded = 0

        for owner, repo in self.CURRICULUM["architecture"]:
            data = self.navigator.navigate_repo(owner, repo)
            if not data:
                continue

            workflow = [
                data["summary"][:50].lower(),
                *data["insights"].get("patterns", [])[:5],
            ]
            hdv = self.hdv_system.encode_workflow(workflow, domain="behavioral")
            self.discovery.record_pattern(
                "behavioral", hdv,
                {
                    "type":    "architecture",
                    "content": f"{owner}/{repo}",
                    "source":  "github",
                },
            )
            self._patterns.append({"source": f"{owner}/{repo}", "type": "architecture"})
            encoded += 1

        return encoded

    def discover_new_capabilities(self) -> List[Dict]:
        """
        Find repos that fill current capability gaps.

        Encodes found repo summaries into the appropriate FICUTS dimension
        and returns the gap discovery list for inspection.
        """
        print("\n[Curriculum] Discovering capability gaps...")
        gaps = self.capability_discovery.discover_repos_for_gaps()

        for gap_info in gaps:
            gap = gap_info["gap"]
            print(f"\n  Gap: {gap}")
            for repo in gap_info["repos"]:
                print(
                    f"    → {repo['owner']}/{repo['name']}"
                    f" (relevance={repo['relevance']:.2f})"
                )
                if repo["relevance"] > 0.0:
                    dim = self.capability_discovery._gap_to_dimension(gap)
                    hdv = self.hdv_system.structural_encode(repo["summary"], dim)
                    self.discovery.record_pattern(
                        dim, hdv,
                        {
                            "type":    gap,
                            "content": repo["summary"][:100],
                            "source":  "capability_discovery",
                        },
                    )

        return gaps

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_patterns(self):
        self.save_path.write_text(json.dumps(self._patterns, indent=2))
        print(
            f"[Curriculum] Saved {len(self._patterns)} patterns"
            f" → {self.save_path}"
        )

    def _load_patterns(self) -> List[Dict]:
        if self.save_path.exists():
            try:
                return json.loads(self.save_path.read_text())
            except Exception:
                pass
        return []

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)
