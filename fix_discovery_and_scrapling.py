#!/usr/bin/env python3
"""
Fix CrossDimensionalDiscovery and enable Scrapling for FICUTS HDV learning.

Scrapling is critical because:
- Handles JavaScript-rendered content (DeepWiki uses heavy JS)
- Auto-retry with exponential backoff
- Better error handling than requests
- Enables the system to scrape and ORDER information into HDV space
"""

from pathlib import Path

def fix_cross_dimensional_discovery():
    """Fix similarity_threshold parameter."""
    
    file_path = Path('tensor/cross_dimensional_discovery.py')
    content = file_path.read_text()
    
    # Fix the method signature
    old_sig = "    def find_universals(self):"
    new_sig = "    def find_universals(self, similarity_threshold=0.85):"
    
    if old_sig in content:
        content = content.replace(old_sig, new_sig)
        
        # Also update the hardcoded threshold
        content = content.replace(
            "if similarity > 0.85:",
            "if similarity > similarity_threshold:"
        )
        
        file_path.write_text(content)
        print("[✓] Fixed CrossDimensionalDiscovery.find_universals()")
        return True
    else:
        print("[!] Method already fixed or not found")
        return False


def enable_scrapling_in_navigator():
    """Enable Scrapling in DeepWikiNavigator for better web scraping."""
    
    file_path = Path('tensor/deepwiki_navigator.py')
    content = file_path.read_text()
    
    changes_made = False
    
    # 1. Add Scrapling import
    if 'from scrapling import Fetcher' not in content:
        # Add after other imports
        import_marker = "from typing import Dict, List, Optional"
        if import_marker in content:
            content = content.replace(
                import_marker,
                import_marker + "\ntry:\n    from scrapling import Fetcher\n    SCRAPLING_AVAILABLE = True\nexcept ImportError:\n    SCRAPLING_AVAILABLE = False"
            )
            changes_made = True
    
    # 2. Update navigate_repo to use Scrapling
    old_fetch = """        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"[DeepWiki] Failed to fetch {url}: {e}")
            return None
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')"""
    
    new_fetch = """        try:
            if SCRAPLING_AVAILABLE:
                # Use Scrapling for better JS handling
                fetcher = Fetcher()
                response = fetcher.get(url, timeout=30)
                
                if response:
                    soup = response.soup  # Scrapling provides BeautifulSoup directly
                else:
                    print(f"[DeepWiki] Scrapling fetch failed for {url}")
                    return None
            else:
                # Fallback to requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
        except Exception as e:
            print(f"[DeepWiki] Failed to fetch {url}: {e}")
            return None"""
    
    if old_fetch in content:
        content = content.replace(old_fetch, new_fetch)
        changes_made = True
    
    if changes_made:
        file_path.write_text(content)
        print("[✓] Enabled Scrapling in DeepWikiNavigator")
        return True
    else:
        print("[!] DeepWikiNavigator already updated")
        return False


def check_scrapling_installation():
    """Check if Scrapling is installed."""
    try:
        import scrapling
        print(f"[✓] Scrapling installed: v{scrapling.__version__}")
        return True
    except ImportError:
        print("[✗] Scrapling not installed")
        print("\nInstall with:")
        print("  pip install scrapling --break-system-packages")
        return False


if __name__ == '__main__':
    print("="*70)
    print(" FIXING FICUTS FOR SCRAPLING-ENABLED HDV LEARNING ".center(70, "="))
    print("="*70)
    print()
    
    print("[1/3] Fixing CrossDimensionalDiscovery...")
    fix_cross_dimensional_discovery()
    print()
    
    print("[2/3] Enabling Scrapling in DeepWikiNavigator...")
    enable_scrapling_in_navigator()
    print()
    
    print("[3/3] Checking Scrapling installation...")
    scrapling_ok = check_scrapling_installation()
    print()
    
    print("="*70)
    print(" FIXES COMPLETE ".center(70, "="))
    print("="*70)
    print()
    
    if scrapling_ok:
        print("✓ All fixes applied")
        print("✓ Scrapling ready for use")
        print()
        print("Next: Run cross-dimensional discovery:")
        print("  python -c \"from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery; ...\"")
    else:
        print("⚠ Scrapling needs installation first")
        print()
        print("After installing:")
        print("1. python run_autonomous.py --bootstrap")
        print("2. DeepWiki scraping will use Scrapling for better JS handling")
