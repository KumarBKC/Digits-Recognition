import os
import re

directory = r"c:\AI.ML WORK\Digits_Recognition"
pattern = re.compile(r"^\s*# -{5,}\s*$")

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            # Skip this script itself just in case
            if "remove_seplines.py" in path:
                continue
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                if not pattern.match(line):
                    new_lines.append(line)
            
            if len(new_lines) != len(lines):
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"Updated {path}")
