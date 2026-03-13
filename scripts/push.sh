#!/bin/bash
# Usage: bash scripts/push.sh <branch> <commit_message>
# Example: bash scripts/push.sh main "fix scheduler bug"

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <branch> <commit_message>"
    echo "Example: $0 main \"fix scheduler bug\""
    exit 1
fi

BRANCH="$1"
COMMENT="$2"
CURRENT_BRANCH=$(git branch --show-current)

git add -A
git commit -m "$COMMENT"

if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git checkout "$BRANCH"
        git merge "$CURRENT_BRANCH"
    else
        git checkout -b "$BRANCH"
    fi
fi

git push -u origin "$BRANCH"

if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    git checkout "$CURRENT_BRANCH"
fi

echo "Done: pushed to origin/$BRANCH with message: $COMMENT"
