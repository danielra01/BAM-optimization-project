#!/bin/bash
# Script to create GitHub issues from the TODO items found in the repository
# This script uses the GitHub CLI (gh) to create issues

set -e

# Check if gh is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Please run: gh auth login"
    exit 1
fi

# Read the JSON file and create issues
ISSUE_FILE="todo_issues.json"

if [ ! -f "$ISSUE_FILE" ]; then
    echo "Error: $ISSUE_FILE not found."
    echo "Please run: python3 create_todo_issues.py first"
    exit 1
fi

# Get the repository (should be danielra01/BAM-optimization-project)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Creating issues in repository: $REPO"
echo ""

# Parse JSON and create issues
python3 << 'EOF'
import json
import subprocess
import sys

with open('todo_issues.json', 'r') as f:
    issues = json.load(f)

created_count = 0
for i, issue in enumerate(issues, 1):
    title = issue['title']
    body = issue['body']
    labels = ','.join(issue['labels'])
    
    print(f"Creating issue {i}/{len(issues)}: {title[:50]}...")
    
    try:
        # Use gh CLI to create the issue
        cmd = [
            'gh', 'issue', 'create',
            '--title', title,
            '--body', body,
            '--label', labels
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issue_url = result.stdout.strip()
        print(f"  ✓ Created: {issue_url}")
        created_count += 1
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to create issue: {e.stderr}")
        sys.exit(1)

print(f"\n✓ Successfully created {created_count}/{len(issues)} issues!")
EOF
