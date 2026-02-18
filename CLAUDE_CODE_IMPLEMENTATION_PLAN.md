# CLAUDE CODE: Curriculum-Based Learning Implementation Plan

## Mission

Implement a system that navigates DeepWiki repos to extract structured learning content, enabling the unified network to:
1. Learn code patterns from challenges (freeCodeCamp)
2. Extract equations + code from books (free-programming-books)
3. Map 3D geometry to G-code (Open3D)
4. Learn architectural patterns (Node.js)
5. Discover repos that add new capabilities to the project

---

## CRITICAL: How DeepWiki Works

DeepWiki is a **web interface** that analyzes GitHub repos. You must:

1. **Navigate web pages** (not API calls)
2. **Parse HTML/JavaScript** (content is rendered client-side)
3. **Extract structured data** from their UI elements

**Example URLs:**
```
https://deepwiki.com/freeCodeCamp/freeCodeCamp
https://deepwiki.com/EbookFoundation/free-programming-books
https://deepwiki.com/isl-org/Open3D
https://deepwiki.com/nodejs/node
```

**What you'll find on each page:**
- Repository summary
- File tree navigation
- Code structure analysis
- Dependency graphs
- Workflow builders (for some repos)

---

## Phase 1: DeepWiki Navigator (NEW - HIGH PRIORITY)

### File: `tensor/deepwiki_navigator.py`

**Purpose:** Navigate DeepWiki web interface, extract structured content

**Implementation:**

```python
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

class DeepWikiNavigator:
    """
    Navigate DeepWiki web interface to extract structured learning content.
    
    DeepWiki analyzes GitHub repos and presents:
    - Repository structure
    - Code summaries
    - Dependency graphs
    - File relationships
    
    We scrape this analysis to build curriculum.
    """
    
    def __init__(self, cache_dir='tensor/data/deepwiki_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = 'https://deepwiki.com'
    
    def navigate_repo(self, owner: str, repo: str) -> Dict:
        """
        Navigate DeepWiki page for a repo.
        
        Args:
            owner: GitHub owner (e.g., 'freeCodeCamp')
            repo: GitHub repo (e.g., 'freeCodeCamp')
        
        Returns:
            {
                'repo_url': 'freeCodeCamp/freeCodeCamp',
                'summary': 'Open source...',
                'file_tree': [...],
                'key_files': [...],
                'dependencies': [...],
                'insights': {...}
            }
        """
        # Check cache first
        cache_file = self.cache_dir / f"{owner}_{repo}.json"
        if cache_file.exists():
            print(f"[DeepWiki] Loading cached: {owner}/{repo}")
            return json.loads(cache_file.read_text())
        
        # Fetch from DeepWiki
        url = f"{self.base_url}/{owner}/{repo}"
        print(f"[DeepWiki] Navigating: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"[DeepWiki] Failed to fetch {url}: {e}")
            return None
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract structured data
        data = {
            'repo_url': f"{owner}/{repo}",
            'summary': self._extract_summary(soup),
            'file_tree': self._extract_file_tree(soup),
            'key_files': self._extract_key_files(soup),
            'dependencies': self._extract_dependencies(soup),
            'insights': self._extract_insights(soup)
        }
        
        # Cache result
        cache_file.write_text(json.dumps(data, indent=2))
        
        return data
    
    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract repository summary from DeepWiki page."""
        # Look for common summary selectors
        selectors = [
            '.repo-summary',
            '.description',
            '#readme',
            'div[data-testid="summary"]'
        ]
        
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.text.strip()
        
        return "No summary found"
    
    def _extract_file_tree(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract file tree structure.
        
        Returns: [
            {'path': 'src/challenges/basic/reverse-string.js', 'type': 'file'},
            {'path': 'src/challenges/advanced/', 'type': 'dir'},
            ...
        ]
        """
        # Look for file tree elements
        file_tree = []
        
        # Common patterns in DeepWiki HTML
        tree_elements = soup.select('.file-tree-item, .file-entry, a[href*="/blob/"]')
        
        for elem in tree_elements:
            path = elem.get('data-path') or elem.text.strip()
            is_dir = 'folder' in elem.get('class', []) or path.endswith('/')
            
            file_tree.append({
                'path': path,
                'type': 'dir' if is_dir else 'file'
            })
        
        return file_tree
    
    def _extract_key_files(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract key files identified by DeepWiki.
        
        DeepWiki often highlights important files (README, main entry points, etc.)
        """
        key_files = []
        
        # Look for highlighted or pinned files
        highlighted = soup.select('.key-file, .important-file, .entry-point')
        
        for elem in highlighted:
            filepath = elem.get('data-path') or elem.text.strip()
            if filepath:
                key_files.append(filepath)
        
        return key_files
    
    def _extract_dependencies(self, soup: BeautifulSoup) -> List[str]:
        """Extract project dependencies."""
        deps = []
        
        # Look for dependency sections
        dep_section = soup.select_one('.dependencies, #dependencies')
        if dep_section:
            dep_items = dep_section.select('li, .dep-item')
            deps = [item.text.strip() for item in dep_items]
        
        return deps
    
    def _extract_insights(self, soup: BeautifulSoup) -> Dict:
        """
        Extract DeepWiki insights (complexity, patterns, etc.).
        
        DeepWiki provides analysis like:
        - Code complexity
        - Common patterns
        - Architecture type
        """
        insights = {}
        
        # Look for insight panels
        complexity = soup.select_one('.complexity-score, [data-metric="complexity"]')
        if complexity:
            insights['complexity'] = complexity.text.strip()
        
        patterns = soup.select('.code-pattern, .pattern-item')
        if patterns:
            insights['patterns'] = [p.text.strip() for p in patterns]
        
        return insights


class DeepWikiChallengeExtractor:
    """
    Extract challenges from freeCodeCamp via DeepWiki.
    
    DeepWiki shows challenge structure. We extract:
    - Challenge descriptions
    - Test cases
    - Solution patterns
    """
    
    def __init__(self, navigator: DeepWikiNavigator):
        self.navigator = navigator
    
    def extract_challenges(self, owner='freeCodeCamp', repo='freeCodeCamp') -> List[Dict]:
        """
        Extract all challenges from freeCodeCamp.
        
        Returns: [
            {
                'id': 'reverse-string',
                'title': 'Reverse a String',
                'difficulty': 'basic',
                'description': '...',
                'test_cases': [...],
                'solution_pattern': '...'
            },
            ...
        ]
        """
        print(f"[ChallengeExtractor] Extracting challenges from {owner}/{repo}")
        
        # Navigate to repo
        repo_data = self.navigator.navigate_repo(owner, repo)
        
        if not repo_data:
            return []
        
        # Find challenge directories
        challenge_dirs = [
            f for f in repo_data['file_tree']
            if 'challenge' in f['path'].lower() and f['type'] == 'dir'
        ]
        
        print(f"[ChallengeExtractor] Found {len(challenge_dirs)} challenge directories")
        
        # Extract challenges from each directory
        challenges = []
        for challenge_dir in challenge_dirs[:10]:  # Limit to first 10 for testing
            challenge = self._extract_challenge_from_dir(owner, repo, challenge_dir['path'])
            if challenge:
                challenges.append(challenge)
        
        # Sort by difficulty
        difficulty_order = {'basic': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        challenges.sort(key=lambda c: difficulty_order.get(c.get('difficulty', 'basic'), 0))
        
        return challenges
    
    def _extract_challenge_from_dir(self, owner: str, repo: str, challenge_path: str) -> Optional[Dict]:
        """Extract single challenge from directory."""
        # Fetch challenge file content from GitHub API
        # (DeepWiki shows structure, GitHub API gives content)
        
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{challenge_path}"
        
        try:
            response = requests.get(api_url)
            if response.status_code != 200:
                return None
            
            files = response.json()
            
            # Look for test file and description
            test_file = None
            desc_file = None
            
            for f in files:
                if 'test' in f['name'].lower():
                    test_file = f
                elif 'readme' in f['name'].lower() or 'description' in f['name'].lower():
                    desc_file = f
            
            if not test_file:
                return None
            
            # Extract challenge data
            challenge_id = Path(challenge_path).name
            difficulty = self._infer_difficulty(challenge_path)
            
            challenge = {
                'id': challenge_id,
                'title': challenge_id.replace('-', ' ').title(),
                'difficulty': difficulty,
                'description': self._fetch_description(desc_file) if desc_file else '',
                'test_cases': self._extract_test_cases(test_file),
                'solution_pattern': 'extracted_pattern'  # Placeholder
            }
            
            return challenge
            
        except Exception as e:
            print(f"[ChallengeExtractor] Error extracting {challenge_path}: {e}")
            return None
    
    def _infer_difficulty(self, path: str) -> str:
        """Infer difficulty from path."""
        path_lower = path.lower()
        if 'basic' in path_lower or 'easy' in path_lower:
            return 'basic'
        elif 'intermediate' in path_lower or 'medium' in path_lower:
            return 'intermediate'
        elif 'advanced' in path_lower or 'hard' in path_lower:
            return 'advanced'
        else:
            return 'basic'
    
    def _fetch_description(self, file_info: Dict) -> str:
        """Fetch file content from GitHub."""
        import base64
        
        try:
            response = requests.get(file_info['url'])
            content_b64 = response.json()['content']
            return base64.b64decode(content_b64).decode('utf-8')
        except:
            return ""
    
    def _extract_test_cases(self, test_file: Dict) -> List[Dict]:
        """Extract test cases from test file."""
        # Parse test file content
        # Extract input/output pairs
        # For now, placeholder
        return [
            {'input': 'example', 'expected': 'output'}
        ]


class DeepWikiBookExtractor:
    """
    Extract programming books curriculum from EbookFoundation/free-programming-books.
    """
    
    def __init__(self, navigator: DeepWikiNavigator):
        self.navigator = navigator
    
    def extract_book_curriculum(self, owner='EbookFoundation', repo='free-programming-books') -> List[Dict]:
        """
        Extract book list organized by topic.
        
        Returns: [
            {
                'title': 'Introduction to Algorithms',
                'authors': ['CLRS'],
                'topic': 'algorithms',
                'url': '...',
                'format': 'PDF'
            },
            ...
        ]
        """
        print(f"[BookExtractor] Extracting books from {owner}/{repo}")
        
        # Navigate repo
        repo_data = self.navigator.navigate_repo(owner, repo)
        
        if not repo_data:
            return []
        
        # Find markdown files (book lists)
        md_files = [
            f for f in repo_data['file_tree']
            if f['path'].endswith('.md') and f['type'] == 'file'
        ]
        
        print(f"[BookExtractor] Found {len(md_files)} markdown files")
        
        # Extract books from each file
        books = []
        for md_file in md_files[:5]:  # Limit for testing
            topic = Path(md_file['path']).stem.lower()
            file_books = self._extract_books_from_md(owner, repo, md_file['path'], topic)
            books.extend(file_books)
        
        return books
    
    def _extract_books_from_md(self, owner: str, repo: str, filepath: str, topic: str) -> List[Dict]:
        """Extract book links from markdown file."""
        # Fetch file content
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}"
        
        try:
            response = requests.get(api_url)
            content_b64 = response.json()['content']
            
            import base64
            content = base64.b64decode(content_b64).decode('utf-8')
            
            # Parse markdown links
            import re
            link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
            matches = re.findall(link_pattern, content)
            
            books = []
            for title, url in matches:
                if any(ext in url.lower() for ext in ['.pdf', '.epub', '.html']):
                    books.append({
                        'title': title,
                        'topic': topic,
                        'url': url,
                        'format': self._infer_format(url)
                    })
            
            return books
            
        except Exception as e:
            print(f"[BookExtractor] Error: {e}")
            return []
    
    def _infer_format(self, url: str) -> str:
        """Infer book format from URL."""
        if '.pdf' in url.lower():
            return 'PDF'
        elif '.epub' in url.lower():
            return 'EPUB'
        elif '.html' in url.lower():
            return 'HTML'
        else:
            return 'Unknown'


class CapabilityDiscovery:
    """
    Discover repos that add capabilities to the project.
    
    Strategy:
    1. Identify current capability gaps
    2. Search DeepWiki for repos that fill gaps
    3. Rank repos by relevance
    4. Extract patterns from top repos
    """
    
    def __init__(self, navigator: DeepWikiNavigator, hdv_system):
        self.navigator = navigator
        self.hdv_system = hdv_system
        self.capability_gaps = self._identify_gaps()
    
    def _identify_gaps(self) -> List[str]:
        """
        Identify what capabilities the project lacks.
        
        Current dimensions:
        - Math: ✓ (equations from papers)
        - Code: ✓ (patterns from GitHub)
        - Physical: ⚠️ (needs more G-code patterns)
        - UI: ⚠️ (needs frontend patterns)
        - Data: ⚠️ (needs data processing patterns)
        
        Returns: List of gap areas
        """
        gaps = [
            'gcode_generation',
            'frontend_frameworks',
            'data_processing',
            'computer_vision',
            'natural_language'
        ]
        
        return gaps
    
    def discover_repos_for_gaps(self) -> List[Dict]:
        """
        Find repos on DeepWiki that fill capability gaps.
        
        Returns: [
            {
                'gap': 'gcode_generation',
                'repos': [
                    {'name': 'PrusaSlicer', 'relevance': 0.95},
                    {'name': 'Cura', 'relevance': 0.92},
                    ...
                ]
            },
            ...
        ]
        """
        discoveries = []
        
        for gap in self.capability_gaps:
            print(f"\n[Discovery] Searching for repos to fill gap: {gap}")
            
            # Search strategy depends on gap type
            repos = self._search_for_gap(gap)
            
            if repos:
                discoveries.append({
                    'gap': gap,
                    'repos': repos
                })
        
        return discoveries
    
    def _search_for_gap(self, gap: str) -> List[Dict]:
        """
        Search for repos relevant to a capability gap.
        
        Strategy:
        1. Use known high-quality repos for each domain
        2. Navigate their DeepWiki pages
        3. Extract patterns
        """
        # Predefined high-quality repos per gap
        gap_repos = {
            'gcode_generation': [
                ('Ultimaker', 'Cura'),
                ('prusa3d', 'PrusaSlicer'),
                ('slic3r', 'Slic3r')
            ],
            'frontend_frameworks': [
                ('facebook', 'react'),
                ('vuejs', 'vue'),
                ('sveltejs', 'svelte')
            ],
            'data_processing': [
                ('pandas-dev', 'pandas'),
                ('numpy', 'numpy'),
                ('apache', 'spark')
            ],
            'computer_vision': [
                ('opencv', 'opencv'),
                ('isl-org', 'Open3D'),
                ('facebookresearch', 'detectron2')
            ],
            'natural_language': [
                ('huggingface', 'transformers'),
                ('explosion', 'spaCy'),
                ('nltk', 'nltk')
            ]
        }
        
        repos_for_gap = gap_repos.get(gap, [])
        
        results = []
        for owner, repo in repos_for_gap[:3]:  # Top 3 per gap
            print(f"  Analyzing: {owner}/{repo}")
            
            # Navigate repo
            data = self.navigator.navigate_repo(owner, repo)
            
            if data:
                # Compute relevance based on description match
                relevance = self._compute_relevance(gap, data['summary'])
                
                results.append({
                    'owner': owner,
                    'name': repo,
                    'relevance': relevance,
                    'summary': data['summary'][:200]
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return results
    
    def _compute_relevance(self, gap: str, summary: str) -> float:
        """
        Compute how relevant a repo is to filling a gap.
        
        Simple version: keyword matching
        Advanced version: HDV similarity
        """
        keywords = {
            'gcode_generation': ['gcode', '3d print', 'slicer', 'toolpath'],
            'frontend_frameworks': ['react', 'component', 'ui', 'frontend'],
            'data_processing': ['dataframe', 'array', 'processing', 'transform'],
            'computer_vision': ['vision', 'image', 'detection', 'recognition'],
            'natural_language': ['nlp', 'language', 'text', 'tokenize']
        }
        
        gap_keywords = keywords.get(gap, [])
        summary_lower = summary.lower()
        
        matches = sum(1 for kw in gap_keywords if kw in summary_lower)
        relevance = matches / len(gap_keywords) if gap_keywords else 0
        
        return relevance
```

**Test:**
```python
def test_deepwiki_navigator():
    navigator = DeepWikiNavigator()
    
    # Test navigation
    data = navigator.navigate_repo('freeCodeCamp', 'freeCodeCamp')
    assert data is not None
    assert 'summary' in data
    assert 'file_tree' in data
    
    print("[PASS] DeepWiki navigation working")

def test_challenge_extraction():
    navigator = DeepWikiNavigator()
    extractor = DeepWikiChallengeExtractor(navigator)
    
    challenges = extractor.extract_challenges()
    assert len(challenges) > 0
    assert challenges[0]['difficulty'] in ['basic', 'intermediate', 'advanced', 'expert']
    
    print(f"[PASS] Extracted {len(challenges)} challenges")

def test_capability_discovery():
    navigator = DeepWikiNavigator()
    hdv_system = IntegratedHDVSystem()
    discovery = CapabilityDiscovery(navigator, hdv_system)
    
    gaps = discovery.discover_repos_for_gaps()
    assert len(gaps) > 0
    assert all('gap' in g and 'repos' in g for g in gaps)
    
    print(f"[PASS] Discovered repos for {len(gaps)} capability gaps")
```

---

## Phase 2: Integrate with Curriculum Trainer

### File: `tensor/curriculum_trainer.py` (UPDATE EXISTING)

**Changes needed:**

```python
class CurriculumTrainer:
    def __init__(self, hdv_system):
        self.hdv_system = hdv_system
        
        # ADD: DeepWiki navigator
        self.navigator = DeepWikiNavigator()
        self.challenge_extractor = DeepWikiChallengeExtractor(self.navigator)
        self.book_extractor = DeepWikiBookExtractor(self.navigator)
        self.capability_discovery = CapabilityDiscovery(self.navigator, hdv_system)
        
        # Keep existing curriculum dict
        self.curriculum = {...}
    
    def _train_freecodecamp(self, repo: str):
        """UPDATE: Use DeepWikiChallengeExtractor instead of placeholder."""
        print(f"[FreeCodeCamp] Extracting challenges via DeepWiki...")
        
        # REPLACE placeholder with actual extraction
        challenges = self.challenge_extractor.extract_challenges()
        
        print(f"[FreeCodeCamp] Found {len(challenges)} challenges")
        
        # Rest of training logic stays the same
        for i, challenge in enumerate(challenges):
            ...
    
    def _train_programming_books(self, repo: str):
        """UPDATE: Use DeepWikiBookExtractor."""
        print(f"[ProgrammingBooks] Extracting books via DeepWiki...")
        
        # REPLACE placeholder
        books = self.book_extractor.extract_book_curriculum()
        
        print(f"[ProgrammingBooks] Found {len(books)} books")
        
        # Rest of training logic stays the same
        for book in books:
            ...
    
    def discover_new_capabilities(self):
        """NEW METHOD: Find repos that add capabilities."""
        print("\n[Discovery] Searching for repos to expand capabilities...")
        
        gaps = self.capability_discovery.discover_repos_for_gaps()
        
        for gap_info in gaps:
            print(f"\nGap: {gap_info['gap']}")
            for repo in gap_info['repos']:
                print(f"  → {repo['owner']}/{repo['name']} (relevance: {repo['relevance']:.2f})")
                
                # Extract patterns from this repo
                repo_data = self.navigator.navigate_repo(repo['owner'], repo['name'])
                
                # Encode patterns to HDV
                if repo_data and repo_data['insights']:
                    for pattern in repo_data['insights'].get('patterns', []):
                        pattern_hdv = self.hdv_system.encode_workflow(pattern, domain='code')
                        # Store in capability library
```

---

## Phase 3: Update run_autonomous.py

### File: `run_autonomous.py` (UPDATE EXISTING)

**Add new command:**

```python
def parse_args():
    parser = argparse.ArgumentParser(...)
    
    # Existing args
    ...
    
    # NEW: Capability discovery
    parser.add_argument('--discover', action='store_true',
                       help='Discover new repos that add capabilities')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Existing logic
    ...
    
    # NEW: Capability discovery
    if args.discover:
        from tensor.curriculum_trainer import CurriculumTrainer
        from tensor.integrated_hdv import IntegratedHDVSystem
        
        hdv_system = IntegratedHDVSystem()
        trainer = CurriculumTrainer(hdv_system)
        
        trainer.discover_new_capabilities()
```

---

## Execution Order

1. **Implement DeepWiki Navigator** (Phase 1)
   ```bash
   # Create tensor/deepwiki_navigator.py
   # Test navigation works
   pytest tests/test_deepwiki_navigator.py
   ```

2. **Integrate with Curriculum Trainer** (Phase 2)
   ```bash
   # Update tensor/curriculum_trainer.py
   # Test challenge extraction
   pytest tests/test_curriculum.py
   ```

3. **Add Capability Discovery** (Phase 3)
   ```bash
   # Update run_autonomous.py
   # Run discovery
   python run_autonomous.py --discover
   ```

4. **Full Pipeline**
   ```bash
   # Populate + Curriculum + Discover + Optimize
   python run_autonomous.py --populate --curriculum --discover --optimize --trials 30
   ```

---

## Expected Outcomes

After implementation:

1. **345 math patterns** (from arXiv papers) ✓ Already have
2. **100+ code patterns** (from freeCodeCamp challenges) ← Phase 1
3. **50+ book curricula** (from free-programming-books) ← Phase 1
4. **20+ 3D patterns** (from Open3D) ← Phase 1
5. **15+ new capability repos** (from discovery) ← Phase 3

**Result:** Network trained on structured curriculum, ready for dev-agent code generation with validated patterns.

---

## File Checklist

- [ ] `tensor/deepwiki_navigator.py` (new)
- [ ] `tests/test_deepwiki_navigator.py` (new)
- [ ] `tensor/curriculum_trainer.py` (update)
- [ ] `tests/test_curriculum.py` (update)
- [ ] `run_autonomous.py` (update)

**Start with Phase 1, Task 1: Implement DeepWikiNavigator class.**

All tests must pass before moving to next phase.

Execute now.
