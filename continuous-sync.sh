#!/bin/bash
# continuous-sync.sh - 持续运行的高频同步守护进程
# 每5分钟检查一次，立即响应WorkBuddy的更新

cd /root/.openclaw/workspace

echo "================================"
echo "🔄 Continuous Sync Daemon Started"
echo "Check interval: 5 minutes"
echo "Started at: $(date)"
echo "================================"

while true; do
    # 获取远程状态
    git fetch origin master --quiet 2>/dev/null
    
    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse origin/master 2>/dev/null)
    
    if [ "$LOCAL" != "$REMOTE" ]; then
        echo ""
        echo "🚀 $(date '+%H:%M:%S') WorkBuddy pushed new code!"
        echo "本地: ${LOCAL:0:7} → 远程: ${REMOTE:0:7}"
        echo "正在拉取并合并..."
        
        if git pull origin master --rebase 2>&1 | tee /tmp/last-pull.log; then
            echo "✅ 同步成功"
            
            # 显示WorkBuddy的提交信息
            echo ""
            echo "WorkBuddy的更新内容："
            git log --oneline --decorate HEAD@{1}..HEAD | head -5
            
            # 记录时间戳
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Synced: $(git log --oneline -1 --pretty=format:'%s')" >> /tmp/continuous-sync.log
            
        else
            echo "⚠️ 同步出现问题，可能需要手动解决"
        fi
        
        echo ""
        echo "================================"
        echo "继续监控... (下次检查: 5分钟后)"
        echo "================================"
    fi
    
    # 等待5分钟
    sleep 300
done
