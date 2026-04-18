# GitHub 仓库配置指南
> AI Evolver 自举系统的远程同步配置

## 步骤 1: 创建 GitHub 仓库

1. 访问 https://github.com/new
2. 填写仓库信息:
   - **Repository name**: `ai-evolver` (或你喜欢的名字)
   - **Description**: AI Evolver 自举系统 - 每2小时的自我进化日志
   - **Public/Private**: 任选 (推荐 Private 除非你想展示进化过程)
   - **Initialize**: ❌ 不要勾选 "Add a README file" (我们已经有内容了)
3. 点击 **Create repository**

## 步骤 2: 获取 GitHub 访问令牌 (PAT)

由于这是自动化系统，推荐使用 Personal Access Token:

1. 访问 https://github.com/settings/tokens
2. 点击 **Generate new token (classic)**
3. 填写信息:
   - **Note**: AI Evolver Auto Push
   - **Expiration**: 选择 "No expiration" (或你偏好的期限)
4. 勾选权限:
   - ✅ **repo** (Full control of private repositories)
5. 点击 **Generate token**
6. **⚠️ 重要**: 立即复制显示的 token (形如 `ghp_xxxxxxx`)，之后无法再次查看！

## 步骤 3: 在本地配置 Git 认证

在终端执行以下命令:

```bash
# 1. 配置远程仓库 (替换 YOUR_USERNAME 和 YOUR_REPO)
cd /root/.openclaw/workspace
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 2. 配置 git 凭证管理器
git config --global credential.helper store

# 3. 首次推送 (会提示输入密码，这里输入你的 PAT)
git push -u origin master
# 用户名: 你的 GitHub 用户名
# 密码: 粘贴刚才复制的 PAT token
```

## 步骤 4: 测试自动进化

运行一次进化脚本来测试 push 是否工作:

```bash
cd /root/.openclaw/workspace
./evolve.sh
```

如果看到 `成功推送到 GitHub!`，说明配置完成！

## 步骤 5: 设置定时任务 (Cron Job)

让系统每2小时自动运行:

```bash
# 编辑 crontab
crontab -e

# 添加这一行 (每2小时执行一次):
0 */2 * * * cd /root/.openclaw/workspace && ./evolve.sh >> /tmp/ai-evolver.log 2>&1

# 保存并退出
```

验证 cron 任务:
```bash
crontab -l
```

## 监控进化过程

### 查看日志
```bash
# 实时查看最后一次进化日志
tail -f /tmp/ai-evolver.log

# 查看进化历史
cat /root/.openclaw/workspace/evolution_log.md
```

### 在 GitHub 上查看
访问你的仓库页面，每2小时会看到新的 commit:
- Commit 消息格式: `evolve: 自动进化循环 #N`
- 可以查看每次进化的详细变化

## 故障排查

### Push 失败 "Authentication failed"
- 检查 PAT token 是否正确
- 确认 token 有 `repo` 权限
- 重新运行 `git push` 并输入正确的凭证

### Cron 不执行
- 检查脚本路径是否正确: `pwd` 显示 /root/.openclaw/workspace
- 检查脚本权限: `chmod +x evolve.sh`
- 手动测试脚本是否能成功运行: `./evolve.sh`

### 想停止自动进化
```bash
# 删除 cron job
crontab -e
# 删除或注释掉 ai-evolver 那一行
```

---
配置完成后，你的 AI Evolver 将每2小时自动:
1. 自我评估和反思
2. 记录进化日志
3. Git commit 所有变化
4. Push 到 GitHub

你可以在 GitHub 上追踪整个进化历程！
