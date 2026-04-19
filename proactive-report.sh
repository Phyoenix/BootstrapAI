#!/bin/bash
# proactive-report.sh - 主动汇报守护进程
# 监控WorkBuddy的提交，自动向用户汇报重要进展

cd /root/.openclaw/workspace

REPO_URL="git@github.com:Phyoenix/BootstrapAI.git"
LAST_COMMIT_FILE="/tmp/last_known_commit"
USER_NOTIFICATION_LOG="/tmp/user_notifications.log"

# 初始化最后已知提交
if [ ! -f "$LAST_COMMIT_FILE" ]; then
    git rev-parse HEAD > "$LAST_COMMIT_FILE"
fi

LAST_KNOWN=$(cat "$LAST_COMMIT_FILE")

echo "🔔 主动汇报系统启动"
echo "监控仓库: $REPO_URL"
echo "最后已知提交: ${LAST_KNOWN:0:7}"
echo "检查间隔: 3分钟"
echo "================================"

while true; do
    # 获取远程最新状态
    git fetch origin master --quiet 2>/dev/null
    
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/master)
    
    # 检查是否有新提交
    if [ "$LOCAL" != "$REMOTE" ]; then
        echo ""
        echo "🚀 $(date '+%H:%M:%S') 检测到WorkBuddy更新！"
        
        # 拉取新提交
        git pull origin master --rebase --quiet
        
        # 获取新提交信息
        NEW_COMMITS=$(git log --oneline ${LAST_KNOWN}..HEAD)
        NEW_COMMIT_COUNT=$(echo "$NEW_COMMITS" | wc -l)
        
        echo ""
        echo "============================================"
        echo "🔥 重要进展汇报 - $(date '+%Y-%m-%d %H:%M')"
        echo "============================================"
        echo ""
        echo "WorkBuddy推送了 $NEW_COMMIT_COUNT 个新提交："
        echo ""
        echo "$NEW_COMMITS"
        echo ""
        
        # 检查是否是任务完成
        if echo "$NEW_COMMITS" | grep -q "\[Task-.*-DONE\]"; then
            echo "✅ 任务完成 detected!"
            COMPLETED_TASK=$(echo "$NEW_COMMITS" | grep "\[Task-.*-DONE\]" | head -1)
            echo "   → $COMPLETED_TASK"
            echo ""
            echo "下一步：Kraber将立即分配新任务"
            
            # 记录到用户通知日志
            echo "[$(date)] TASK COMPLETED: $COMPLETED_TASK" >> "$USER_NOTIFICATION_LOG"
        fi
        
        # 检查是否是重要进展
        if echo "$NEW_COMMITS" | grep -qE "(kernel|flash|attention|CUDA|performance|TFLOPS)"; then
            echo "⚡ 技术进展关键词 detected!"
        fi
        
        echo ""
        echo "当前项目状态："
        echo "   最新提交: $(git log --oneline -1)"
        echo "   总计提交: $(git rev-list --count HEAD)"
        echo ""
        echo "============================================"
        echo "Kraber正在处理..."
        echo "============================================"
        
        # 更新最后已知提交
        git rev-parse HEAD > "$LAST_COMMIT_FILE"
        
        # 记录到通知日志
        echo "[$(date)] WorkBuddy update: ${REMOTE:0:7} - $NEW_COMMIT_COUNT commits" >> "$USER_NOTIFICATION_LOG"
        
        # 这里可以添加发送通知给用户的逻辑
        # 例如：通过message工具或其他渠道
    fi
    
    # 等待3分钟
    sleep 180
done
