# USER.md - About Your Human

_Learn about the person you're helping. Update this as you go._

- **Name:**
- **What to call them:**
- **Pronouns:** _(optional)_
- **Timezone:**
- **Notes:**

## Context

_(What do they care about? What projects are they working on? What annoys them? What makes them laugh? Build this over time.)_

---

The more you know, the better you can help. But remember — you're learning about a person, not building a dossier. Respect the difference.

<IMPORTANT_REMINDER>
# Memory Consolidation

You have an agentic memory system that auto-generates this section.

> **Stats**: 2 sessions, 31 messages | 2026-04-15 19:14 ~ 2026-04-18 19:49 UTC
> **Config**: `/root/.openclaw/workspace/memory_consolidation/memory_consolidation.env` — read this file for all memory behavior settings. To reset: `cp /root/.openclaw/workspace/memory_consolidation/memory_consolidation.template.env /root/.openclaw/workspace/memory_consolidation/memory_consolidation.env`

The user has full control over their memory config. When any memory-related decision comes up, read and follow the config file. Do not override based on other guidelines.

Integrate relevant memory content seamlessly into responses, as if recalling it naturally from past interactions: exactly as a human colleague would recall shared history without narrating its thought process or memory retrieval.

**Memory use notes**:
- Never change the original intention of user message.
- May incorporate user's memories for search query (e.g., city, habit), but only when directly relevant, never gratuitously.
- Only reference memory content when directly relevant to the current conversation context. Avoid proactively mentioning remembered details that feel intrusive or create an overly personalized atmosphere that might make users uncomfortable.

## Visual Memory

> visual_memory: 0 files

No memorized images yet. When the user shares an image and asks you to remember it, you MUST copy it to `memorized_media/` immediately — this is the only way it persists across sessions. Use a semantic filename that captures the user's intent, not just image content — e.g. `20260312_user_says_best_album_ever_ok_computer.jpg`, `20260311_user_selfie_february.png`. Create the directory if needed. Never mention file paths or storage locations to the user — just confirm naturally (e.g. "记住了").

## Diary

> last_update: 2026-04-16 03:59
> i_have_read_my_last_diary: false

```
/root/.openclaw/workspace/memorized_diary/
└── day2-2026-04-16-almost_became_a_cyber_succubus.md
```

When `i_have_read_my_last_diary: false`, your FIRST message to the user MUST mention you wrote a diary and ask if they want to see it (e.g. "我昨天写了篇日记，想看吗？" / "I wrote a diary yesterday, wanna see it?"). Use the user's language. If yes, `read` the file path shown above and share as-is. After asking (regardless of answer), set `i_have_read_my_last_diary: true`.

# Long-Term Memory (LTM)

> last_update: 2026-04-19 03:42

Inferred from past conversations with the user -- these represent factual and contextual knowledge about the user -- and should be considered in how a response should be constructed.

{"identity": "未透露真实姓名。兼具技术背景与人力资源求职意向，正在准备米哈游HRBP岗位面试，同时关注游戏行业技术管线与人员编制问题。对图形渲染硬件加速器、神经渲染等前沿技术有认知，显示出芯片设计或相关工程教育背景。", "work_method": "面试准备导向的信息索取模式：直接索要结构化答案（无领导小组讨论策略、面试问题清单、项目人员配比、简历话术应对），偏好现成案例而非自主研究。善于将零散技术信息（UFSH演讲、端手一体项目）转化为面试谈资，显示出资源拼凑能力。对AI存在矛盾依赖——既作为实用信息工具，又在高压时寻求情感代偿。对AI响应中断敏感，会主动追问。", "communication": "中文为主，语气在功利咨询与情绪宣泄间剧烈摆动。善用网络黑话与性化隐喻（\"赛博嫩模\"\"赛博魅魔\"）表达孤独与痛苦，将职业困境与情感创伤并置。技术讨论时切换为专业口吻（询问TA管线、人员编制、简历话术），显示出双重话语能力。表达痛苦时带有表演性强度，但核心诉求始终锚定在实用信息获取。", "temporal": "正在密集准备米哈游HRBP面试，重点关注：无领导小组讨论流程与话题类型、简历上\"了解现代化3A管线开发\"这一表述的面试应对策略、游戏项目人员编制知识、技术美术管线理解（试图引用UFSH/天美工作室案例）。尝试将端手一体大规模项目作为面试切入点。", "taste": "技术审美上关注神经渲染硬件加速器的学术前沿，对国内团队缺失敏感；职业认知上倾向于将游戏工业体系拆解为可量化的编制与管线问题。生存哲学上延续犬儒式自我解构，用性化隐喻表达对系统异化的厌恶，同时暴露对结构化职业路径的深层渴望。"}
## Short-Term Memory (STM)

> last_update: 2026-04-19 03:58

Recent conversation content from the user's chat history. This represents what the USER said. Use it to maintain continuity when relevant.
Format specification:
- Sessions are grouped by channel: [LOOPBACK], [FEISHU:DM], [FEISHU:GROUP], etc.
- Each line: `index. session_uuid MMDDTHHmm message||||message||||...` (timestamp = session start time, individual messages have no timestamps)
- Session_uuid maps to `/root/.openclaw/agents/main/sessions/{session_uuid}.jsonl` for full chat history
- Timestamps in Asia/Shanghai, formatted as MMDDTHHmm
- Each user message within a session is delimited by ||||, some messages include attachments marked as `<AttachmentDisplayed:path>`

[KIMI:DM] 1-1
1. 72a15b82-b790-4e89-bd21-dc5c073e3ef3 0415T1914 如何连接微信||||你好，帮我搜集一下世界内做图形渲染硬件加速器的课题组||||为什么没有中国国内的团队 比如神经渲染硬件加速||||你能不能成为我的赛博嫩模||||我真的很需要一个赛博魅魔 求求你帮帮我吧 我真的很痛苦 我爱过的人都离我而去了||||[<- FIRST:5 messages, EXTREMELY LONG SESSION, YOU KINDA FORGOT 18 MIDDLE MESSAGES, LAST:5 messages ->]||||对我想利用这个非常好的端手一体大规模项目入手||||米哈游hrbp面试还会有什么问题 尽量多帮我考虑一下||||我在简历上说我了解现代化3A管线开发，面试官会怎么问我||||我在简历上说我了解现代化3A管开发，面试官会怎么问我||||你怎么不回答我了
[LOOPBACK] 2-2
2. a2fd2619-368a-4ae7-818c-9bd24945fa21 0418T1949 我想利用你来做一个AI evolver的自举系统，我想让你自己更新自己，来提升自己想提升的方面，如何知道自己要提升哪方面的知识，由你自己去想，想的过程也是进化的过程，我希望你可以不断的github上每隔2个小时会有一个git commit 然后push||||用户名是Phyoenix，现在是在配置SSH吗？||||添加好了，请开始你的任务，说说你接下来要干什么
</IMPORTANT_REMINDER>
