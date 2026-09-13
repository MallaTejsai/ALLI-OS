"""One-off repair: replace literal 0x08 backspace bytes in query_understanding.py
with proper \\\\b word-boundary sequences in regex raw strings."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/memory/foundation/query_understanding.py"

with open(path, "rb") as f:
    data = f.read()

count = data.count(b"\x08")
if count == 0:
    print("No backspace bytes found - nothing to do")
    sys.exit(0)

# 0x08 -> backslash + 'b'  (the two characters that spell a regex word boundary)
fixed = data.replace(b"\x08", b"\\b")

with open(path, "wb") as f:
    f.write(fixed)

print(f"Replaced {count} backspace bytes in {path}")

# Verify
with open(path, "rb") as f:
    check = f.read()
print("Remaining 0x08 bytes:", check.count(b"\x08"))
