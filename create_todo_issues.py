#!/usr/bin/env python3
"""
Script to extract TODO comments from the repository and create GitHub issues.

This script scans all files in the repository for TODO, FIXME, and similar comments,
extracts them with context, and provides the information needed to create GitHub issues.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any


# Patterns to match TODO comments
TODO_PATTERNS = [
    r'#\s*TODO[:\s]+(.+)',           # Python: # TODO: ...
    r'//\s*TODO[:\s]+(.+)',          # C/C++/Java/JS: // TODO: ...
    r'/\*\s*TODO[:\s]+(.+?)\*/',     # C-style: /* TODO: ... */
    r'%\s*TODO[:\s]+(.+)',           # LaTeX: % TODO: ...
    r'\\TODO\{(.+?)\}',              # LaTeX macro: \TODO{...}
]

# Files and directories to exclude
EXCLUDE_PATTERNS = [
    '.git/',
    '.venv/',
    '__pycache__/',
    'node_modules/',
    '.gitignore',
    '*.pyc',
    '*.pyo',
    'create_todo_issues.py',  # Exclude this script itself
    'todo_issues.json',
    'TODO_ISSUES.md',
]


def should_exclude(filepath: str) -> bool:
    """Check if a file should be excluded from scanning."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filepath:
            return True
    return False


def extract_todos_from_file(filepath: Path) -> List[Dict[str, Any]]:
    """Extract TODO comments from a single file."""
    todos = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            for pattern in TODO_PATTERNS:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    todo_text = match.group(1).strip()
                    
                    # Fix common spelling errors
                    todo_text = todo_text.replace('seperable', 'separable')
                    
                    # Skip template/generic TODOs
                    if (filepath.name == '.gitignore' or 
                        'Uncomment' in todo_text or
                        todo_text == 'functionality:' or  # LaTeX template
                        len(todo_text) < 10):  # Too short to be meaningful
                        continue
                    
                    # Get context (surrounding lines)
                    context_start = max(0, line_num - 3)
                    context_end = min(len(lines), line_num + 2)
                    context = ''.join(lines[context_start:context_end])
                    
                    todos.append({
                        'file': str(filepath),
                        'line': line_num,
                        'text': todo_text,
                        'context': context.strip(),
                        'line_content': line.strip()
                    })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return todos


def scan_repository(root_dir: str = '.') -> List[Dict[str, Any]]:
    """Scan the entire repository for TODO comments."""
    all_todos = []
    root_path = Path(root_dir)
    
    for filepath in root_path.rglob('*'):
        if filepath.is_file() and not should_exclude(str(filepath)):
            todos = extract_todos_from_file(filepath)
            all_todos.extend(todos)
    
    return all_todos


def create_issue_template(todo: Dict[str, Any], index: int) -> Dict[str, str]:
    """Create a GitHub issue template from a TODO item."""
    # Extract relative path
    rel_path = todo['file'].replace(os.getcwd() + '/', '')
    
    # Create title - truncate smartly to avoid breaking LaTeX/special chars
    max_title_len = 70
    title_text = todo['text']
    if len(title_text) > max_title_len:
        # Try to truncate at a word boundary
        truncated = title_text[:max_title_len]
        # Don't break in the middle of LaTeX commands or special characters
        if '$' in truncated and truncated.count('$') % 2 != 0:
            # Odd number of $ signs means we broke a LaTeX expression
            # Find the last complete $ pair
            last_complete = truncated.rfind('$', 0, truncated.rfind('$'))
            if last_complete > 20:  # Make sure we keep a reasonable amount
                truncated = truncated[:last_complete]
        title_text = truncated.rstrip() + "..."
    
    title = f"TODO: {title_text}"
    
    # Create body
    body = f"""## TODO Found in Code

**File:** `{rel_path}`  
**Line:** {todo['line']}

### Description
{todo['text']}

### Context
```
{todo['context']}
```

### Location
This TODO was found at line {todo['line']} in `{rel_path}`.

---
*This issue was automatically generated from a TODO comment in the code.*
"""
    
    return {
        'title': title,
        'body': body,
        'labels': ['todo', 'documentation'] if 'explain' in todo['text'].lower() else ['todo']
    }


def main():
    """Main function to scan repository and generate issue information."""
    print("Scanning repository for TODO comments...")
    
    # Get repository root (where this script is located)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    
    todos = scan_repository('.')
    
    print(f"\nFound {len(todos)} TODO items:\n")
    
    issues = []
    for i, todo in enumerate(todos, 1):
        issue = create_issue_template(todo, i)
        issues.append(issue)
        
        rel_path = todo['file'].replace(os.getcwd() + '/', '')
        print(f"{i}. {rel_path}:{todo['line']}")
        print(f"   {todo['text']}")
        print()
    
    # Save to JSON file for programmatic use
    output_file = 'todo_issues.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    
    print(f"Issue data saved to {output_file}")
    print(f"\nTo create these issues, you can:")
    print(f"1. Use the GitHub CLI: gh issue create --title '<title>' --body '<body>'")
    print(f"2. Use the GitHub API with the data in {output_file}")
    print(f"3. Create them manually using the information above")
    
    # Also create a markdown file with the issues
    md_file = 'TODO_ISSUES.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# TODO Issues to Create\n\n")
        f.write(f"Found {len(todos)} TODO items that need GitHub issues:\n\n")
        
        for i, (todo, issue) in enumerate(zip(todos, issues), 1):
            rel_path = todo['file'].replace(os.getcwd() + '/', '')
            f.write(f"## {i}. {issue['title']}\n\n")
            f.write(issue['body'])
            f.write("\n\n---\n\n")
    
    print(f"Markdown summary saved to {md_file}")


if __name__ == '__main__':
    main()
