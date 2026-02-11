# TODO to GitHub Issues - Documentation

This directory contains tools to automatically find TODO comments in the repository and create GitHub issues for them.

## What Was Done

The following scripts and files have been created to help manage TODO items:

1. **`create_todo_issues.py`** - Python script that scans the repository for TODO comments
2. **`create_issues.sh`** - Bash script that creates GitHub issues using the GitHub CLI
3. **`todo_issues.json`** - JSON file containing structured data for all TODO items found
4. **`TODO_ISSUES.md`** - Markdown file with human-readable summary of all TODOs

## TODOs Found

The scripts identified **3 TODO items** in the repository:

1. **File:** `report/mathematical-background.tex`, **Line:** 82
   - Explain what is happening in the seperable case from a geometric point of view!

2. **File:** `report/mathematical-background.tex`, **Line:** 88
   - explain why regularization is a good thing to do!

3. **File:** `report/mathematical-background.tex`, **Line:** 90
   - It is fairly obvious that the optimization problem is $C^\infty$. Where should we put this?

## How to Create the Issues

### Method 1: Using the Automated Script (Recommended)

If you have the GitHub CLI (`gh`) installed and authenticated:

```bash
# First, ensure gh CLI is installed
# Visit: https://cli.github.com/

# Authenticate with GitHub
gh auth login

# Run the script to scan for TODOs and generate issue data
python3 create_todo_issues.py

# Create all the issues automatically
./create_issues.sh
```

### Method 2: Manual Creation

You can also create the issues manually using the information in `TODO_ISSUES.md`:

1. Go to https://github.com/danielra01/BAM-optimization-project/issues/new
2. For each TODO item in `TODO_ISSUES.md`:
   - Copy the title
   - Copy the body
   - Add labels: `todo` and optionally `documentation`
   - Click "Submit new issue"

### Method 3: Using GitHub API

If you prefer to use the GitHub API directly, the `todo_issues.json` file contains all the structured data you need:

```bash
# Using curl (requires a GitHub personal access token)
TOKEN="your_github_token"
REPO="danielra01/BAM-optimization-project"

# Read each issue from the JSON file and create it
# (See GitHub API documentation for details)
```

## Re-scanning for TODOs

If new TODOs are added to the codebase, you can re-run the scan:

```bash
python3 create_todo_issues.py
```

This will update the `todo_issues.json` and `TODO_ISSUES.md` files with any new TODOs found.

## Files Generated

- **`todo_issues.json`** - Machine-readable JSON with issue data
- **`TODO_ISSUES.md`** - Human-readable markdown summary
- These files are automatically updated each time `create_todo_issues.py` is run

## Customization

### Excluding Files or Patterns

Edit the `EXCLUDE_PATTERNS` list in `create_todo_issues.py` to exclude specific files or directories from scanning.

### Adding New TODO Patterns

Edit the `TODO_PATTERNS` list in `create_todo_issues.py` to recognize different comment styles or keywords.

## Notes

- The script automatically excludes `.git`, `.venv`, `__pycache__`, and other common directories
- Template TODOs (like those in `.gitignore`) are automatically filtered out
- The script provides context (surrounding lines) for each TODO to help understand what needs to be done
