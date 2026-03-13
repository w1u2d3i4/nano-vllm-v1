---
name: push
description: Stage all changes, commit, and push to a specified branch. Usage: /push <branch> <commit_message>
argument-hint: <branch> <commit_message>
allowed-tools: Bash
---

# Push to Remote

Stage all changes, commit with the given message, and push to the specified branch.

## Arguments

- `$ARGUMENTS[0]` — target branch name
- `$ARGUMENTS[1]` — commit message

## Steps

1. Run `git add -A` to stage all changes
2. Run `git commit -m "$ARGUMENTS[1]"`
3. If target branch differs from current:
   - If branch exists locally: checkout and merge
   - If branch doesn't exist: create it from current branch
4. Run `git push -u origin $ARGUMENTS[0]`
5. Switch back to original branch if needed

## Implementation

```bash
cd /opt/data/private/llm_test/nano-vllm-v1
BRANCH="$ARGUMENTS[0]"
COMMENT="$ARGUMENTS[1]"
CURRENT=$(git branch --show-current)

git add -A
git commit -m "$COMMENT"

if [ "$CURRENT" != "$BRANCH" ]; then
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git checkout "$BRANCH"
        git merge "$CURRENT"
    else
        git checkout -b "$BRANCH"
    fi
fi

git push -u origin "$BRANCH"

if [ "$CURRENT" != "$BRANCH" ]; then
    git checkout "$CURRENT"
fi
```
