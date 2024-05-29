"""
Answer: NO
        9, 10, 17 also sem to be popular
"""

import sys
from pathlib import Path
import hickle as hkl

def visit(the_dir):
    the_dir = the_dir.resolve()
    for child in the_dir.iterdir():
        go(child)

def check(hkl_path):
    boh = hkl.load(hkl_path).astype(int)
    if boh.shape[0] != 15:
        print(f"ZAN ZAN ZAN {boh.shape[0]}")

def go(path):
    if path.is_dir():
        visit(path)
    elif path.suffix == ".hkl":
        check(path)
    # else pass

start = Path(sys.argv[1]).resolve()
go(start)
