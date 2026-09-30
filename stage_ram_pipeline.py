#!/usr/bin/env python3
"""
stage_ram_pipeline.py
Stages fly connectome CSR binary streams and receptor weights into POSIX shared memory (/dev/shm).
Eliminates disk I/O during high-throughput GPU simulation loops.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path

SOURCE_DIR = Path("/home/<user>/Projects/science/fly-connectome/data")
RAM_DIR = Path("/dev/shm/fly-connectome/data")

REQUIRED_FILES = [
    "row_ptr.bin",
    "col_idx.bin",
    "weights.bin",
    "neuron_nt_types.bin",
    "descending_indices.bin",
    "neuron_receptor_sensitivities.bin"
]

def hash_file(path: Path) -> str:
    """Computes fast SHA256 of first 1MB and last 1MB for verification."""
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size <= 2 * 1024 * 1024:
            h.update(f.read())
        else:
            h.update(f.read(1024 * 1024))
            f.seek(size - 1024 * 1024)
            h.update(f.read(1024 * 1024))
    return h.hexdigest()

def stage_to_ram(verbose: bool = True) -> Path:
    """Copies required connectome binary streams into /dev/shm."""
    RAM_DIR.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    
    for filename in REQUIRED_FILES:
        src = SOURCE_DIR / filename
        dst = RAM_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required source file: {src}")
            
        # Copy only if destination missing or size mismatch
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            if verbose:
                print(f"[RAM-STAGE] Copied {filename} ({src.stat().st_size / 1024 / 1024:.2f} MB)")
        total_bytes += dst.stat().st_size
        
    if verbose:
        print(f"[RAM-STAGE] Staging complete: {len(REQUIRED_FILES)} files, {total_bytes / 1024 / 1024:.2f} MB in {RAM_DIR}")
    return RAM_DIR

def verify_ram_stage() -> bool:
    """Verifies byte and hash parity between disk and /dev/shm."""
    if not RAM_DIR.exists():
        print(f"[VERIFY-FAIL] {RAM_DIR} does not exist.")
        return False
        
    all_ok = True
    for filename in REQUIRED_FILES:
        src = SOURCE_DIR / filename
        dst = RAM_DIR / filename
        if not dst.exists():
            print(f"[VERIFY-FAIL] Missing in RAM: {filename}")
            all_ok = False
            continue
        if src.stat().st_size != dst.stat().st_size:
            print(f"[VERIFY-FAIL] Size mismatch {filename}: src={src.stat().st_size} dst={dst.stat().st_size}")
            all_ok = False
            continue
        if hash_file(src) != hash_file(dst):
            print(f"[VERIFY-FAIL] Checksum mismatch {filename}")
            all_ok = False
            continue
        print(f"[VERIFY-OK] {filename}: {dst.stat().st_size / 1024 / 1024:.2f} MB verified")
        
    return all_ok

def get_data_dir() -> Path:
    """Returns RAM_DIR if staged, otherwise falls back to SOURCE_DIR."""
    if RAM_DIR.exists() and all((RAM_DIR / f).exists() for f in REQUIRED_FILES):
        return RAM_DIR
    return SOURCE_DIR

if __name__ == "__main__":
    if "--verify" in sys.argv:
        ok = verify_ram_stage()
        sys.exit(0 if ok else 1)
    else:
        stage_to_ram(verbose=True)
        ok = verify_ram_stage()
        sys.exit(0 if ok else 1)
