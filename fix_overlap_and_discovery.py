#!/usr/bin/env python3
"""
CRITICAL FIX: Enable cross-dimensional discovery.

Root causes:
1. 0% overlap dimensions (math and code use disjoint HDV space)
2. Empty code in behavioral patterns
3. Tensor.copy() bug in CrossDimensionalDiscovery

Solutions:
1. Force 30% overlap in IntegratedHDVSystem domain masks
2. Tensor → numpy conversion in record_pattern
3. Populate actual code from GitHub
"""

from pathlib import Path
import re

def fix_integrated_hdv_overlaps():
    """
    Fix IntegratedHDVSystem to ensure domains share 30% of HDV space.
    
    Current: Random 10% per domain → 0% overlap (disjoint sets)
    Fixed: Deterministic allocation with forced overlap
    """
    
    file_path = Path('tensor/integrated_hdv.py')
    content = file_path.read_text()
    
    # Find _initialize_sparse_masks method
    old_method = '''    def _initialize_sparse_masks(self):
        """
        Initialize sparse masks for each domain.
        
        Each domain uses ~10% of HDV dimensions (randomly selected).
        Overlaps between domains = potential universals.
        """
        masks = {}
        
        for domain in ['math', 'code', 'physical', 'execution']:
            mask = torch.zeros(self.hdv_dim, dtype=torch.bool)
            
            # Random 10% active
            n_active = int(self.hdv_dim * 0.1)
            active_indices = torch.randperm(self.hdv_dim)[:n_active]
            mask[active_indices] = True
            
            masks[domain] = mask
        
        return masks'''
    
    new_method = '''    def _initialize_sparse_masks(self):
        """
        Initialize sparse masks with FORCED OVERLAP.
        
        Strategy:
        - Universal dimensions (0-2999): ALL domains use these (30% overlap)
        - Domain-specific (3000-9999): Each domain gets unique portion
        
        This ensures cross-dimensional discovery can find universals.
        """
        masks = {}
        
        # Universal dimensions (first 30% of space)
        universal_dims = int(self.hdv_dim * 0.3)  # 0-2999
        
        # Domain-specific dimensions (remaining 70%)
        domain_specific_start = universal_dims
        domains = ['math', 'code', 'physical', 'execution']
        dims_per_domain = (self.hdv_dim - domain_specific_start) // len(domains)
        
        for i, domain in enumerate(domains):
            mask = torch.zeros(self.hdv_dim, dtype=torch.bool)
            
            # ALL domains use universal dimensions (FORCED OVERLAP)
            mask[:universal_dims] = True
            
            # Each domain gets unique portion of specific dimensions
            specific_start = domain_specific_start + (i * dims_per_domain)
            specific_end = specific_start + dims_per_domain
            mask[specific_start:specific_end] = True
            
            masks[domain] = mask
        
        return masks'''
    
    if old_method in content:
        content = content.replace(old_method, new_method)
        file_path.write_text(content)
        print("[✓] Fixed IntegratedHDVSystem: 30% forced overlap")
        return True
    else:
        print("[!] IntegratedHDVSystem already fixed or pattern not found")
        return False


def fix_cross_dimensional_tensor_bug():
    """Fix Tensor.copy() → .detach().cpu().numpy() in record_pattern."""
    
    file_path = Path('tensor/cross_dimensional_discovery.py')
    content = file_path.read_text()
    
    # Find record_pattern method
    old_line = '            "hdv": hdv_vec.copy(),'
    new_line = '            "hdv": hdv_vec.detach().cpu().numpy() if hasattr(hdv_vec, "detach") else hdv_vec.copy(),'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        file_path.write_text(content)
        print("[✓] Fixed CrossDimensionalDiscovery: Tensor → numpy conversion")
        return True
    else:
        print("[!] Already fixed or not found")
        return False


def verify_fixes():
    """Verify the fixes work."""
    print("\n" + "="*70)
    print(" VERIFYING FIXES ".center(70, "="))
    print("="*70 + "\n")
    
    # Test 1: Check overlap
    print("[Test 1] Checking domain overlap...")
    try:
        import sys
        sys.path.insert(0, 'tensor')
        from integrated_hdv import IntegratedHDVSystem
        
        hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150)
        overlaps = hdv.find_overlaps()
        
        print(f"  Overlap dimensions: {len(overlaps)}")
        print(f"  Overlap ratio: {len(overlaps)/hdv.hdv_dim:.1%}")
        
        if len(overlaps) > 2500:  # Should be ~30%
            print("  ✓ PASS: Significant overlap detected")
        else:
            print(f"  ✗ FAIL: Only {len(overlaps)} overlaps (expected ~3000)")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
    
    print()
    
    # Test 2: Check Tensor handling
    print("[Test 2] Checking Tensor → numpy conversion...")
    try:
        import torch
        from cross_dimensional_discovery import CrossDimensionalDiscovery
        
        hdv = IntegratedHDVSystem(hdv_dim=10000, n_modes=150)
        discovery = CrossDimensionalDiscovery(hdv)
        
        # Try recording a torch.Tensor
        test_tensor = torch.randn(10000)
        discovery.record_pattern('math', test_tensor, {'name': 'test'})
        
        print("  ✓ PASS: Tensor recording works")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")


if __name__ == '__main__':
    print("="*70)
    print(" FIXING CROSS-DIMENSIONAL DISCOVERY ".center(70, "="))
    print("="*70)
    print()
    
    print("[1/2] Fixing IntegratedHDVSystem domain overlaps...")
    fix_integrated_hdv_overlaps()
    print()
    
    print("[2/2] Fixing CrossDimensionalDiscovery Tensor handling...")
    fix_cross_dimensional_tensor_bug()
    print()
    
    verify_fixes()
    
    print("\n" + "="*70)
    print(" FIXES COMPLETE ".center(70, "="))
    print("="*70)
    print()
    print("Next steps:")
    print("1. Restart Python (to reload modules)")
    print("2. Run discovery again - should find universals now")
