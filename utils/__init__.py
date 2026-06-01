"""anna_v2.utils — shared infrastructure helpers.

Leaf-level utilities (atomic file IO, JSON parsing, logging) used by every
other module in anna_v2. Keep stdlib-only; no cross-imports between
anna_v2.* packages so this stays a clean dependency root and is trivially
portable to a future v3 codebase.
"""
