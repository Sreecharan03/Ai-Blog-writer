"""
Replace all non-ASCII em-dashes and arrows with Python Unicode escapes.
This makes the file 100% ASCII-safe while preserving the characters semantically.
"""
import os

files = [
    "app/api/article_run.py",
    "app/api/article_revise.py",
    "app/api/article_qc.py",
    "app/api/article_zerogpt_fix.py",
    "app/services/article_graph.py",
]

# Raw byte sequences -> ASCII-safe replacement bytes
replacements = [
    # mangled em-dash (UTF-8 of â + € + right-dq) -> hyphen-hyphen
    (b'\xc3\xa2\xe2\x82\xac\xe2\x80\x9d', b' - '),
    # proper em-dash (UTF-8 U+2014) -> hyphen-hyphen
    (b'\xe2\x80\x94', b' - '),
    # mangled right-arrow (UTF-8 of â + †+ ') -> ASCII arrow
    (b'\xc3\xa2\xe2\x80\xa0\x92', b'->'),
    # proper right arrow UTF-8 U+2192
    (b'\xe2\x86\x92', b'->'),
    # mangled right-single-quote -> apostrophe
    (b'\xc3\xa2\xe2\x82\xac\xe2\x84\xa2', b"'"),
    # proper right single quote U+2019
    (b'\xe2\x80\x99', b"'"),
    # proper left single quote U+2018
    (b'\xe2\x80\x98', b"'"),
    # proper left double quote U+201C
    (b'\xe2\x80\x9c', b'"'),
    # proper right double quote U+201D
    (b'\xe2\x80\x9d', b'"'),
    # en-dash U+2013
    (b'\xe2\x80\x93', b' - '),
    # ellipsis U+2026
    (b'\xe2\x80\xa6', b'...'),
]

for path in files:
    if not os.path.exists(path):
        print("SKIP:", path)
        continue
    with open(path, "rb") as f:
        raw = f.read()
    fixed = raw
    total = 0
    for bad, good in replacements:
        count = fixed.count(bad)
        if count:
            fixed = fixed.replace(bad, good)
            print("  %d x %r -> %r  in %s" % (count, bad, good, path))
            total += count
    if fixed != raw:
        with open(path, "wb") as f:
            f.write(fixed)
        print("FIXED (%d replacements): %s" % (total, path))
    else:
        print("clean: %s" % path)

print("\nDone. Verifying syntax...")
import py_compile, sys
for path in files:
    if not os.path.exists(path):
        continue
    try:
        py_compile.compile(path, doraise=True)
        print("  OK:", path)
    except py_compile.PyCompileError as e:
        print("  FAIL:", path, str(e))
