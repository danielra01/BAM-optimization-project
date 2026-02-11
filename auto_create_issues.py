#!/usr/bin/env python3
"""
Automated GitHub Issue Creator for TODO Comments

This script creates GitHub issues directly using the GitHub API for all TODO
comments found in the repository.

Usage:
    # With GitHub token from environment
    export GITHUB_TOKEN="your_token_here"
    python3 auto_create_issues.py
    
    # Or provide token as argument
    python3 auto_create_issues.py --token your_token_here
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


def create_github_issue(owner, repo, title, body, labels, token):
    """Create a GitHub issue using the API."""
    url = f'https://api.github.com/repos/{owner}/{repo}/issues'
    
    data = json.dumps({
        'title': title,
        'body': body,
        'labels': labels
    }).encode('utf-8')
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
    }
    
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['html_url']
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode('utf-8')
        raise Exception(f"Failed to create issue: {e.code} - {error_msg}")


def main():
    parser = argparse.ArgumentParser(
        description='Create GitHub issues from TODO comments'
    )
    parser.add_argument(
        '--token',
        help='GitHub personal access token (or set GITHUB_TOKEN env var)',
        default=os.environ.get('GITHUB_TOKEN')
    )
    parser.add_argument(
        '--owner',
        help='Repository owner',
        default='danielra01'
    )
    parser.add_argument(
        '--repo',
        help='Repository name',
        default='BAM-optimization-project'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without actually creating issues'
    )
    
    args = parser.parse_args()
    
    if not args.token and not args.dry_run:
        print("Error: GitHub token required.")
        print("Either set GITHUB_TOKEN environment variable or use --token flag")
        print("\nTo create a token:")
        print("1. Go to https://github.com/settings/tokens")
        print("2. Click 'Generate new token (classic)'")
        print("3. Select scope: 'repo' (Full control of private repositories)")
        print("4. Generate and copy the token")
        sys.exit(1)
    
    # Load the issues from JSON file
    json_file = 'todo_issues.json'
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found")
        print("Run: python3 create_todo_issues.py first")
        sys.exit(1)
    
    with open(json_file, 'r') as f:
        issues = json.load(f)
    
    print(f"Found {len(issues)} TODO items to create as issues")
    print(f"Repository: {args.owner}/{args.repo}")
    print()
    
    if args.dry_run:
        print("DRY RUN MODE - No issues will be created")
        print()
    
    created_urls = []
    for i, issue in enumerate(issues, 1):
        title = issue['title']
        body = issue['body']
        labels = issue['labels']
        
        print(f"[{i}/{len(issues)}] {title[:60]}...")
        
        if args.dry_run:
            print(f"  Would create with labels: {', '.join(labels)}")
        else:
            try:
                url = create_github_issue(
                    args.owner,
                    args.repo,
                    title,
                    body,
                    labels,
                    args.token
                )
                print(f"  ✓ Created: {url}")
                created_urls.append(url)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                sys.exit(1)
    
    print()
    if args.dry_run:
        print(f"Dry run complete. {len(issues)} issues would be created.")
        print("Remove --dry-run flag to actually create the issues.")
    else:
        print(f"✓ Successfully created {len(created_urls)} issues!")
        if created_urls:
            print("\nCreated issues:")
            for url in created_urls:
                print(f"  - {url}")


if __name__ == '__main__':
    main()
