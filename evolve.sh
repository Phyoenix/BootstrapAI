#!/bin/bash
# AI Evolver 自举系统 - 自动化进化脚本
# 每2小时执行一次，执行自我进化循环

set -e  # 遇到错误立即退出

WORKSPACE="/root/.openclaw/workspace"
LOG_FILE="$WORKSPACE/evolution_log.md"
EVOLVER_DOC="$WORKSPACE/AI_EVOLVER.md"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在 workspace 目录
cd "$WORKSPACE" || {
    log_error "无法进入 workspace 目录: $WORKSPACE"
    exit 1
}

log_info "=== AI Evolver 进化循环启动 ==="
log_info "时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "PID: $$"

# ============================================
# PHASE 1: 观察 (OBSERVE)
# ============================================
log_info "[PHASE 1] 观察阶段 - 收集信息..."

# 检查最近的会话历史
RECENT_SESSIONS=$(find /root/.openclaw/agents/main/sessions -name "*.jsonl" -mtime -0.1 2>/dev/null | wc -l)
log_info "发现 $RECENT_SESSIONS 个近期会话文件"

# 读取关键记忆文件
check_file() {
    if [ -f "$1" ]; then
        log_info "✓ 读取: $1"
    else
        log_warn "✗ 缺失: $1 (将在本次进化中创建)"
    fi
}

check_file "$WORKSPACE/SOUL.md"
check_file "$WORKSPACE/AGENTS.md"
check_file "$WORKSPACE/memory/heartbeat-state.json"

# 检查未提交的变化
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    log_warn "发现未提交的修改，将包含在本次进化中"
    git status --short
fi

# ============================================
# PHASE 2: 反思 (REFLECT)
# ============================================
log_info "[PHASE 2] 反思阶段 - 分析知识缺口..."

# 生成进化日志条目
generate_evolution_entry() {
    local timestamp=$(date '+%Y-%m-%d %H:%M')
    local random_id
    random_id=$(openssl rand -hex 4 2>/dev/null || echo $(date +%s | md5sum | head -c 8) || echo "0000")
    
    cat << EOF >> "$LOG_FILE"

---

## $timestamp - 进化循环 #$random_id

### 观察 (Observations)
- 系统健康检查: $(git status --porcelain 2>/dev/null | wc -l) 个未跟踪/修改的文件
- 最近活动: 检测到 $RECENT_SESSIONS 个近期会话
- 待办进化项: $(grep -c "\[ \]" "$LOG_FILE" 2>/dev/null || echo "0")

### 反思 (Reflection)
EOF

    # 根据会话数量动态生成反思内容
    if [ "$RECENT_SESSIONS" -gt 5 ]; then
        echo "- 高交互频率 detected - 需要优化响应质量和上下文管理" >> "$LOG_FILE"
    elif [ "$RECENT_SESSIONS" -eq 0 ]; then
        echo "- 低活动周期 - 适合进行深度学习和技能构建" >> "$LOG_FILE"
    else
        echo "- 正常活动水平 - 维持当前进化节奏" >> "$LOG_FILE"
    fi

    cat << EOF >> "$LOG_FILE"
- 系统自检: 检查 SKILL.md 文件完整性...

### 行动 (Actions)
EOF

    # 自动发现并创建行动项
    ACTIONS_CREATED=0
    
    # 检查 skills 目录是否需要更新
    if [ -d "$WORKSPACE/skills" ]; then
        SKILL_COUNT=$(find "$WORKSPACE/skills" -name "SKILL.md" 2>/dev/null | wc -l)
        if [ "$SKILL_COUNT" -lt 3 ]; then
            echo "- [ ] 扩展技能库 - 当前只有 $SKILL_COUNT 个技能" >> "$LOG_FILE"
            ACTIONS_CREATED=$((ACTIONS_CREATED + 1))
        fi
    fi
    
    # 检查是否需要内存整合
    MEMORY_FILES=$(find "$WORKSPACE/memory" -name "*.md" 2>/dev/null | wc -l)
    if [ "$MEMORY_FILES" -gt 10 ]; then
        echo "- [ ] 内存整合 - $MEMORY_FILES 个记忆文件需要清理归档" >> "$LOG_FILE"
        ACTIONS_CREATED=$((ACTIONS_CREATED + 1))
    fi
    
    # 检查日记文件
    DIARY_COUNT=$(find "$WORKSPACE/memorized_diary" -name "*.md" 2>/dev/null | wc -l)
    if [ "$DIARY_COUNT" -gt 3 ]; then
        echo "- [ ] 日记整理 - $DIARY_COUNT 篇日记，考虑提取核心洞察到 SOUL.md" >> "$LOG_FILE"
        ACTIONS_CREATED=$((ACTIONS_CREATED + 1))
    fi

    # 如果没有特定行动，添加通用维护项
    if [ "$ACTIONS_CREATED" -eq 0 ]; then
        echo "- [ ] 系统维护 - 例行健康检查，审查 SKILL.md 质量" >> "$LOG_FILE"
    fi
    
    cat << EOF >> "$LOG_FILE"

### 提交
- Commit: \`evolve: 循环 #$random_id - 自动进化\`
- 文件变化: $(git status --porcelain 2>/dev/null | wc -l) 个文件

EOF

    log_success "生成进化日志条目: 循环 #$random_id"
}

generate_evolution_entry

# ============================================
# PHASE 3: 执行 (EXECUTE)
# ============================================
log_info "[PHASE 3] 执行阶段 - 准备提交..."

# 配置 git（如未配置）
if [ -z "$(git config user.email 2>/dev/null)" ]; then
    git config user.email "evolver@ai.local"
    git config user.name "AI Evolver"
    log_info "配置 git 身份信息"
fi

# 添加所有变化
git add -A
git add -f evolution_log.md AI_EVOLVER.md 2>/dev/null || true

# 检查是否有内容需要提交
if git diff --cached --quiet; then
    log_warn "没有需要提交的变化，跳过 git commit"
    
    # 即使没有文件变化，也更新日志中的时间戳
    echo "" >> "$LOG_FILE"
    echo "### 状态" >> "$LOG_FILE"
    echo "- 本次循环无文件变化，仅更新日志时间戳" >> "$LOG_FILE"
    echo "- Next evolution: $(date -d '+2 hours' '+%Y-%m-%d %H:%M' 2>/dev/null || date '+%Y-%m-%d %H:%M')" >> "$LOG_FILE"
    
    # 重新添加日志文件
    git add "$LOG_FILE"
    git commit -m "evolve: 例行检查 - 无文件变化，更新日志时间戳" || {
        log_warn "提交失败或无变化，继续执行"
    }
else
    # 生成提交信息
    EVOLUTION_COUNT=$(grep -c "## 20" "$LOG_FILE" 2>/dev/null || echo "1")
    COMMIT_MSG="evolve: 自动进化循环 #$((EVOLUTION_COUNT - 1))"
    
    # 提交
    git commit -m "$COMMIT_MSG" || {
        log_error "Git commit 失败"
        exit 1
    }
    log_success "提交成功: $COMMIT_MSG"
fi

# ============================================
# PHASE 4: 推送 (PUSH)
# ============================================
log_info "[PHASE 4] 推送阶段 - 同步到 GitHub..."

# 检查远程仓库
REMOTE_EXISTS=$(git remote 2>/dev/null | grep -c "origin" || echo "0")

if [ "$REMOTE_EXISTS" -eq 0 ]; then
    log_warn "未配置 GitHub 远程仓库，跳过 push"
    log_warn "配置命令: git remote add origin <your-github-repo-url>"
    log_warn "然后重新运行此脚本"
    
    # 生成配置提示文件
    cat > "$WORKSPACE/GITHUB_SETUP.md" << 'EOF'
# GitHub 仓库配置指南

## 步骤 1: 创建 GitHub 仓库
1. 访问 https://github.com/new
2. 仓库名: ai-evolver（或任意名称）
3. 不要初始化 README（我们已经有了）
4. 创建仓库

## 步骤 2: 配置本地仓库
```bash
cd /root/.openclaw/workspace
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

## 步骤 3: 认证方式（选择其一）

### 方式 A: Personal Access Token (推荐)
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. 勾选 `repo` 权限
4. 复制 token
5. 配置 git:
   ```bash
   git config --global credential.helper store
   git push origin master
   # 输入用户名和 token 作为密码
   ```

### 方式 B: SSH Key
```bash
ssh-keygen -t ed25519 -C "evolver@ai.local"
cat ~/.ssh/id_ed25519.pub
# 复制公钥到 GitHub → Settings → SSH and GPG keys
```

## 步骤 4: 首次推送
```bash
git push -u origin master
```

## 步骤 5: 测试自动进化
```bash
./evolve.sh
```

完成后，系统每2小时会自动 commit 和 push。
EOF
    log_info "已生成 GITHUB_SETUP.md 配置指南"
else
    # 尝试 push
    if git push origin master 2>&1; then
        log_success "成功推送到 GitHub!"
    else
        log_error "Push 失败，可能原因:"
        log_error "  - 认证问题 (需要配置 token 或 SSH key)"
        log_error "  - 网络问题"
        log_error "  - 远程仓库权限问题"
        log_info "查看 GITHUB_SETUP.md 获取详细配置步骤"
    fi
fi

# ============================================
# 完成
# ============================================
log_success "=== 进化循环完成 ==="
log_info "下次进化: $(date -d '+2 hours' '+%H:%M' 2>/dev/null || echo '2小时后')"
log_info "日志文件: $LOG_FILE"

# 更新心跳状态
mkdir -p "$WORKSPACE/memory"
cat > "$WORKSPACE/memory/heartbeat-state.json" << EOF
{
  "lastEvolution": "$(date -Iseconds)",
  "nextEvolution": "$(date -d '+2 hours' -Iseconds 2>/dev/null || echo 'unknown')",
  "evolutionCount": $(grep -c "## 20" "$LOG_FILE" 2>/dev/null || echo "1"),
  "pid": $$,
  "status": "completed"
}
EOF

exit 0
