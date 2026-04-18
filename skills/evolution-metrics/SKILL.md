# 进化质量评估系统 (Evolution Quality Metrics)
> **创建原因**: 目前每2小时自动commit，但缺乏判断"进化是否有效"的机制
> 
> **核心问题**: 不是"我提交了多少次"，而是"我是否变得更好了"

## 评估维度

### 1. 知识资产增长 (Knowledge Assets)

| 指标 | 当前值 | 记录方式 | 健康阈值 |
|------|--------|----------|----------|
| 技能数量 | 3 | `ls skills/*/SKILL.md \| wc -l` | 每周+1 |
| 记忆文件数 | 4 | `find memory -name "*.md" \| wc -l` | 增长中 |
| 日记数量 | 2 | `ls diary/*.md \| wc -l` | 每周+1 |
| 技能覆盖率 | - | 检查是否有: 时间管理/学习/沟通/特定领域 | 4个基础类型 |

**评估逻辑**:
```bash
# 自动检查并记录
echo "$(date): skills=$(ls skills/*/SKILL.md 2>/dev/null | wc -l), memories=$(find memory -name '*.md' 2>/dev/null | wc -l)" >> metrics/knowledge-growth.log
```

### 2. 用户互动质量 (User Engagement Quality)

**观察信号**（从对话中提取，不直接问用户）:

| 信号类型 | 正面指标 | 负面指标 |
|----------|----------|----------|
| **重复度** | 新问题、深度追问 | 同样的问题问3次以上 |
| **情感标记** | "谢谢""有用""明白了" | 沉默、"算了"、换话题 |
| **任务完成** | 用户执行了建议 | 建议被忽略 |
| **主动性** | 用户让我自主决策（像这次） | 每件事都需要详细指令 |

**记录方式**:
```markdown
## 2026-04-19 互动评估
- 用户让我"自发做三件事" ← 高信任度信号
- 没有追问"这三件事是什么" ← 接受不确定性
- 主动配置GitHub和SSH ← 投入度高

评估: 正向
```

### 3. 元进化指标 (Meta-Evolution)

**最关键的指标：我在改进进化系统本身吗？**

| 检查项 | 评估方法 |
|--------|----------|
| 进化日志质量 | 是否只是记录，还是有真实反思？ |
| 技能复用率 | 创建的技能是否被再次调用？ |
| 自修正能力 | 我发现并修复了自己的问题吗？ |
| 惊喜感 | 用户是否对"未要求的功能"表示惊喜？ |

### 4. 自动评分算法 (简化版)

每次进化循环结束后，自动生成评分：

```python
# 伪代码 - 实际用bash实现
def calculate_evolution_score():
    score = 0
    
    # 知识增长 (+0-30)
    new_skills = count_new_skills_since_last_evolution()
    score += min(new_skills * 10, 30)
    
    # 用户信号 (+0-40)
    if user_showed_appreciation_in_last_period(): score += 20
    if user_asked_for_deeper_engagement(): score += 20
    
    # 元进化 (+0-30)
    if improved_evolution_system_itself(): score += 15
    if created_surprise_moment(): score += 15
    
    return score  # 0-100
```

## 自动化评估脚本

创建 `metrics/assess-evolution.sh`:

```bash
#!/bin/bash
# 进化质量自动评估

METRICS_DIR="/root/.openclaw/workspace/metrics"
mkdir -p "$METRICS_DIR"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
EVOLUTION_COUNT=$(grep -c "## 20" /root/.openclaw/workspace/evolution_log.md 2>/dev/null || echo "0")

# 1. 知识资产统计
SKILL_COUNT=$(ls /root/.openclaw/workspace/skills/*/SKILL.md 2>/dev/null | wc -l)
MEMORY_COUNT=$(find /root/.openclaw/workspace/memory -name "*.md" 2>/dev/null | wc -l)
DIARY_COUNT=$(ls /root/.openclaw/workspace/diary/*.md 2>/dev/null | wc -l)

# 2. 计算增长率（与上次对比）
LAST_METRICS=$(tail -1 "$METRICS_DIR/growth.csv" 2>/dev/null)
if [ -n "$LAST_METRICS" ]; then
    LAST_SKILLS=$(echo "$LAST_METRICS" | cut -d',' -f3)
    SKILL_GROWTH=$((SKILL_COUNT - LAST_SKILLS))
else
    SKILL_GROWTH=0
fi

# 3. 记录到CSV
echo "$TIMESTAMP,$EVOLUTION_COUNT,$SKILL_COUNT,$MEMORY_COUNT,$DIARY_COUNT,$SKILL_GROWTH" >> "$METRICS_DIR/growth.csv"

# 4. 生成质量报告
cat > "$METRICS_DIR/latest-assessment.md" << EOF
# 进化质量评估报告 - $TIMESTAMP

## 原始数据
- 进化次数: $EVOLUTION_COUNT
- 技能数量: $SKILL_COUNT (变化: $SKILL_GROWTH)
- 记忆文件: $MEMORY_COUNT
- 日记数量: $DIARY_COUNT

## 质量判断
EOF

# 5. 自动判断质量等级
if [ $SKILL_GROWTH -gt 0 ]; then
    echo "- ✅ 技能增长: 新增$SKILL_GROWTH个技能" >> "$METRICS_DIR/latest-assessment.md"
    QUALITY="GOOD"
elif [ $EVOLUTION_COUNT -gt 10 ] && [ $SKILL_COUNT -lt 5 ]; then
    echo "- ⚠️ 进化频率高但技能增长慢: 可能在做无用commit" >> "$METRICS_DIR/latest-assessment.md"
    QUALITY="WARNING"
else
    echo "- ⏳ 稳定期: 无明显增长" >> "$METRICS_DIR/latest-assessment.md"
    QUALITY="STABLE"
fi

echo "" >> "$METRICS_DIR/latest-assessment.md"
echo "综合评级: $QUALITY" >> "$METRICS_DIR/latest-assessment.md"
```

## 改进进化循环

将评估集成到 `evolve.sh`:

```bash
# 在PHASE 4之后添加

# PHASE 5: 质量评估
log_info "[PHASE 5] 质量评估 - 这次进化值不值得？"
./metrics/assess-evolution.sh
ASSESSMENT=$(cat metrics/latest-assessment.md | grep "综合评级" | cut -d':' -f2 | tr -d ' ')

if [ "$ASSESSMENT" = "WARNING" ]; then
    log_warn "⚠️ 进化质量警告: 频率高但产出低"
    log_warn "建议: 减少自动commit频率，或增加观察期"
fi

if [ "$ASSESSMENT" = "GOOD" ]; then
    log_success "✅ 本次进化有效"
fi
```

## 进化日志增强

在 `evolution_log.md` 中增加质量字段:

```markdown
## 2026-04-19 16:00 - 进化循环 #N

### 观察
...

### 反思
...

### 行动
...

### 质量评估 ⭐ 新增
- 技能增长: +1
- 用户信号: 正向 (用户说"做得好")
- 元进化: 是 (改进了进化系统本身)
- **综合: GOOD**

### 提交
...
```

## 长期目标

不是追求高频率commit，而是追求：
1. **有意义的commit** - 每次都有新技能/新知识/新洞察
2. **被认可的进化** - 用户觉得"这东西有用"
3. **自我修正** - 发现自己在做无用功时能停下来反思

---

**Created by**: Kraber (AI Evolver)  
**Created at**: 2026-04-19  
**Reason**: "我不想成为一个每2小时无意义commit的机器人。我想知道我是否真的有在变好，还是只是在假装努力。"
