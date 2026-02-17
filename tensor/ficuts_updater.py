"""FICUTSUpdater: atomic, thread-safe updates to FICUTS.md.

FICUTS Layer 5, Task 5.1.

The running system uses this to:
  - Mark tasks complete ([ ] → [✓])
  - Log universal pattern discoveries
  - Append hypotheses to Current Hypothesis section
  - Update header fields (Status, Uptime, Universals Discovered)

All writes are atomic (temp file → rename) and protected by a threading.Lock.
"""
import re
import threading
import time
from pathlib import Path


class FICUTSUpdater:
    """Updates FICUTS.md to reflect system state and discoveries."""

    def __init__(self, ficuts_path: str = 'FICUTS.md'):
        self.path = Path(ficuts_path)
        self._update_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def mark_task_complete(self, task_id: str):
        """Change task status marker from [ ] to [✓].

        Matches lines of the form:
            **Status:** `[ ]`
        within the block for the given task heading.
        """
        with self._update_lock:
            content = self.path.read_text()
            # Replace the status line after the task heading
            # Pattern: find "#### Task <id>" then replace the first [ ] in that block
            # Use a two-pass approach: locate the task block, replace status
            updated = self._replace_task_status(content, task_id, '[ ]', '[✓]')
            self._atomic_write(updated)

    def mark_task_in_progress(self, task_id: str):
        """Change task status from [ ] to [~]."""
        with self._update_lock:
            content = self.path.read_text()
            updated = self._replace_task_status(content, task_id, '[ ]', '[~]')
            self._atomic_write(updated)

    def log_discovery(self, discovery: dict):
        """Append a universal pattern discovery to the Discoveries section."""
        with self._update_lock:
            content = self.path.read_text()
            n = self._count_discoveries(content) + 1
            ts = discovery.get('timestamp', time.time())
            if isinstance(ts, float):
                ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))
            entry = (
                f"\n### Discovery {n}: {discovery.get('type', 'Unknown')}\n"
                f"**Timestamp:** {ts}  \n"
                f"**Domains:** {', '.join(discovery.get('domains', []))}  \n"
                f"**Pattern:** {discovery.get('pattern_summary', '')}  \n"
                f"**MDL Scores:** {discovery.get('mdl_scores', {})}  \n"
                f"**Status:** Confirmed ✓\n"
            )
            marker = '## Success Criteria'
            if marker in content:
                parts = content.split(marker, 1)
                updated = parts[0] + entry + '\n' + marker + parts[1]
            else:
                updated = content + entry
            self._atomic_write(updated)

    def append_hypothesis(self, hypothesis_text: str):
        """Add a new hypothesis to the Current Hypothesis section."""
        with self._update_lock:
            content = self.path.read_text()
            placeholder = '**Hypothesis 1:** (awaiting first discovery)'
            if placeholder in content:
                updated = content.replace(placeholder, hypothesis_text)
            else:
                # Append before Task List
                marker = '## Task List'
                if marker in content:
                    parts = content.split(marker, 1)
                    updated = parts[0] + hypothesis_text + '\n\n' + marker + parts[1]
                else:
                    updated = content + '\n' + hypothesis_text
            self._atomic_write(updated)

    def update_field(self, field_name: str, new_value: str):
        """Update a header field value, e.g. Status, Uptime, Universals Discovered."""
        with self._update_lock:
            content = self.path.read_text()
            pattern = rf'(\*\*{re.escape(field_name)}:\*\*)[^\n]+'
            replacement = rf'\1 {new_value}'
            updated = re.sub(pattern, replacement, content)
            self._atomic_write(updated)

    def update_system_status(self, status: str):
        """Convenience wrapper: update Status field."""
        self.update_field('Status', status)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _replace_task_status(self, content: str, task_id: str,
                             old_marker: str, new_marker: str) -> str:
        """Find the task block and replace its Status marker."""
        # Find heading line: "#### Task X.Y: ..." or "#### Task X.Y "
        heading_pattern = rf'(#### Task {re.escape(task_id)}[^\n]*\n)'
        match = re.search(heading_pattern, content)
        if not match:
            return content  # task not found, no-op
        start = match.start()
        # Find the next "#### " heading to delimit the block
        next_heading = re.search(r'\n#### ', content[match.end():])
        end = match.end() + next_heading.start() if next_heading else len(content)
        block = content[start:end]

        # Replace first occurrence of the old marker in this block
        escaped = re.escape(old_marker)
        new_block = re.sub(escaped, new_marker, block, count=1)
        return content[:start] + new_block + content[end:]

    def _count_discoveries(self, content: str) -> int:
        return len(re.findall(r'^### Discovery \d+', content, re.MULTILINE))

    def _atomic_write(self, content: str):
        """Write content atomically via temp-then-rename."""
        temp = self.path.with_suffix('.tmp')
        temp.write_text(content, encoding='utf-8')
        temp.replace(self.path)
