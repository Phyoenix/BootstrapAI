#!/bin/bash
# sync.sh - 3小时同步脚本 (Kraber & WorkBuddy协作)
# 功能: 拉取最新代码,提交本地更改,推送到GitHub

set -e

cd /root/.openclaw/workspace

echo "================================"
echo "🔄 Git Sync - $(date '+%Y-%m-%d %H:%M')"
echo "================================"

# 检查是否有WorkBuddy的锁文件
if [ -f ".sync-in-progress" ]; then
    echo "⏳ WorkBuddy正在同步,本次跳过"
    exit 0
fi

# 创建自己的锁文件
touch ".sync-in-progress-kraber"

# 先拉取最新更改
echo "📥 Pulling from origin..."
if git pull origin master --rebase 2>&1 | tee /tmp/git-pull.log; then
    echo "✅ Pull successful"
    
    # 检查WorkBuddy的新提交
    NEW_COMMITS=$(git log --oneline HEAD@{1}..HEAD 2>/dev/null | wc -l)
    if [ "$NEW_COMMITS" -gt 0 ]; then
        echo "🔍 WorkBuddy提交了 $NEW_COMMITS 个新commit:"
        git log --oneline HEAD@{1}..HEAD
    fi
else
    echo "⚠️ Pull had issues, checking status..."
    git status --short
fi

# 检查本地更改
if [ -n "$(git status --porcelain)" ]; then
    echo "📤 Local changes detected, committing..."
    git add -A
    git commit -m "sync: auto-update $(date '+%m-%d %H:%M') - Kraber" || true
    
    echo "📤 Pushing to origin..."
    if git push origin master 2>&1 | tee /tmp/git-push.log; then
        echo "✅ Push successful"
    else
        echo "⚠️ Push failed, will retry next sync"
    fi
else
    echo "ℹ️ No local changes to commit"
fi

# 移除锁文件
rm -f ".sync-in-progress-kraber"

echo "================================"
echo "✅ Sync complete - $(date '+%H:%M')"
echo "Next sync in 3 hours"
echo "================================"
