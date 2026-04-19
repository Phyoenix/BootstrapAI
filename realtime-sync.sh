#!/bin/bash
# realtime-sync.sh - 实时高频同步脚本
# 每10分钟检查一次WorkBuddy的更新，立即响应

set -e

cd /root/.openclaw/workspace

echo "🔄 Real-time Sync Check - $(date '+%H:%M:%S')"

# 获取远程最新状态
git fetch origin master --quiet

# 检查是否有新提交
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/master)

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "📥 WorkBuddy有新提交！立即拉取..."
    git pull origin master --rebase 2>&1
    
    # 显示WorkBuddy的更新
    echo "🔍 WorkBuddy的最新提交："
    git log --oneline HEAD@{1}..HEAD
    
    # 记录到日志
    echo "[$(date '+%H:%M:%S')] Pulled WorkBuddy updates: $(git log --oneline -1)" >> /tmp/realtime-sync.log
    
    # 如果有用户指定的响应脚本，执行它
    if [ -f ".on-workbuddy-update.sh" ]; then
        bash .on-workbuddy-update.sh
    fi
else
    echo "✓ 无新提交"
fi
