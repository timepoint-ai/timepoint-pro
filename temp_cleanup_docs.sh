#!/bin/bash
# Documentation Cleanup Script for Timepoint-Daedalus
# Removes outdated documentation files and keeps only essential docs

set -e  # Exit on error

echo "=========================================="
echo "Timepoint-Daedalus Documentation Cleanup"
echo "=========================================="
echo ""

# Safety check - make sure we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "MECHANICS.md" ]; then
    echo "❌ Error: README.md or MECHANICS.md not found"
    echo "   Are you in the timepoint-daedalus directory?"
    exit 1
fi

echo "📋 Current documentation files:"
ls -1 *.md 2>/dev/null || echo "No .md files found"
echo ""

# Confirm deletion
echo "⚠️  This script will DELETE the following files:"
echo "   - CHANGE-ROUND.md (replaced by PLAN.md)"
echo "   - CURRENT_STATE_ANALYSIS.md (outdated)"
echo "   - IMPLEMENTATION-PROOF.md (outdated)"
echo "   - ORCHESTRATOR-GUIDE.md (merged into README)"
echo "   - README_TESTING.md (redundant)"
echo ""
echo "The following will be KEPT:"
echo "   ✅ README.md"
echo "   ✅ MECHANICS.md"
echo "   ✅ PLAN.md"
echo "   ✅ TESTING.md (will need manual update)"
echo ""

read -p "Continue with cleanup? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo ""
echo "🗑️  Removing outdated documentation..."

# Function to safely remove file
safe_remove() {
    if [ -f "$1" ]; then
        rm "$1"
        echo "   ✅ Removed: $1"
    else
        echo "   ⏭️  Not found: $1 (already removed?)"
    fi
}

# Remove outdated docs
safe_remove "CHANGE-ROUND.md"
safe_remove "CURRENT_STATE_ANALYSIS.md"
safe_remove "IMPLEMENTATION-PROOF.md"
safe_remove "ORCHESTRATOR-GUIDE.md"
safe_remove "README_TESTING.md"

echo ""
echo "✨ Cleanup complete!"
echo ""
echo "📄 Remaining documentation:"
ls -1 *.md 2>/dev/null || echo "No .md files found"
echo ""

echo "📝 Next steps:"
echo "1. Review the new README.md, MECHANICS.md, and PLAN.md"
echo "2. Update TESTING.md with current test status (manual)"
echo "3. Commit changes:"
echo "   git add README.md MECHANICS.md PLAN.md"
echo "   git add -u  # Stage deletions"
echo "   git commit -m 'docs: Clean up outdated documentation'"
echo ""
echo "🎉 Done!"
