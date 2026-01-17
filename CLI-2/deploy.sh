#!/bin/bash
# Deploy to GitHub - Run this to push to HIMANSHUMOURYADTU/CLI

echo "🚀 DataClean Pro - GitHub Deployment Script"
echo "==========================================="
echo ""

# Step 1: Configure GitHub (if not already done)
echo "✓ Configuring Git..."
git config user.name "Himanshu Moury" 2>/dev/null || true
git config user.email "himanshumoury@example.com" 2>/dev/null || true

# Step 2: Show current status
echo "✓ Current git status:"
git log --oneline | head -5
echo ""

# Step 3: Check if remote exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "✓ Remote already configured:"
    git remote -v
else
    echo "⚠ Remote not configured. To push to GitHub, run:"
    echo "  git remote add origin https://github.com/HIMANSHUMOURYADTU/CLI.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
    exit 1
fi

echo ""
echo "✓ Repository ready for GitHub!"
echo ""
echo "Next steps:"
echo "1. Create new repo at: https://github.com/new"
echo "2. Name it: CLI"
echo "3. Leave it EMPTY"
echo "4. Run: git remote add origin https://github.com/HIMANSHUMOURYADTU/CLI.git"
echo "5. Run: git branch -M main && git push -u origin main"
echo ""
