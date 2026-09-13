# Git 使用指南（针对本项目）

> 写给：初次使用 git 的你
> 目标：能安全地"看历史、对比改动、回退版本"，且**永不丢工作**

---

## 一、你的项目 git 现状（先搞清楚）

```bash
$ git log --oneline
1fa37ee Gohr Neural Distinguisher PyTorch 复现: SPECK32加密+数据生成+BitSlicedResNet训练
```

**只有 1 个提交**（项目最初的版本）。而之后做的所有事情——包括**那个关键的输入表示 bug 修复**——都还没提交。

查看当前有哪些未提交改动：
```bash
git status          # 简洁版：git status -s
```

你现在的状态大致是：

| 类型 | 文件 |
|---|---|
| 已修改（M） | README.md、docs/复现学习记录.md、requirements.txt、src/model/distinguisher.py、train.py |
| 未跟踪（??） | plot_log.py、docs/experiments/、各 bat/py 脚本、各说明文档 |

---

## 二、⚠️ 第一条铁律：回溯之前，先提交

**git 的"未提交改动"是脆弱的——很多回溯命令会直接丢掉它们，且无法恢复。**

所以，**做任何回溯操作前，先执行**：

```bash
git add -A
git commit -m "保存当前进度：修复输入表示bug + 补充文档与工具脚本"
```

提交之后：
- 你的工作被永久记录（随时能找回）
- 回溯操作变得**安全**（因为改动已经在历史里了）

> 记住这句话：**"提交"不是"上传"**，它只是把当前状态记到本地历史里，随时可以做。
> 提交多少次都不丢人——**丢工作才丢人**。

---

## 三、日常操作（5 个命令够用 90%）

| 命令 | 作用 |
|---|---|
| `git status` | 看当前有哪些改动 |
| `git add -A` | 把所有改动"暂存"（准备提交） |
| `git commit -m "说明"` | 提交，形成一个历史节点 |
| `git log --oneline` | 看历史（每个提交一行） |
| `git diff` | 看未暂存改动的具体内容（逐行 +/-） |

**典型工作流**：
```bash
git status                      # 看看改了啥
git add -A                      # 全部暂存
git commit -m "修复: xxx"        # 提交
git log --oneline               # 确认已记录
```

**查看某个文件的改动历史**：
```bash
git log --oneline -- src/model/distinguisher.py
```

---

## 四、回溯操作（4 种场景，按危险程度排序）

### 场景 1：只想"看看"历史版本（最安全）

```bash
git show <提交号>                    # 看某次提交改了什么
git show <提交号>:README.md          # 看某次提交时某文件的内容
```

> 提交号就是 `git log --oneline` 里那串 `1fa37ee` 这样的字符。

### 场景 2：临时切到某个历史状态看看，然后回来（安全）

```bash
git checkout 1fa37ee        # 切到那个提交的状态（此时是"分离头指针"状态）
# ... 随便看、随便试 ...
git switch -                # 回到最新状态（或 git checkout main）
```

⚠️ 在这种状态下**不要提交**（容易搞乱历史），看完就切回来。

### 场景 3：只撤销某个文件的改动（危险！会丢该文件的工作）

```bash
git restore README.md                # 把 README.md 恢复到上次提交的状态
git restore src/model/distinguisher.py
```

⚠️ **被 restore 的文件改动会永久消失**（除非它们已提交过）。
> 用 `git diff <文件>` 先看清楚你要丢掉什么。

### 场景 4：整个项目回退到某个提交（极度危险）

```bash
git reset --hard 1fa37ee     # 回到初始提交，之后所有改动全部消失
```

⚠️⚠️ **这是最危险的命令**。执行前务必：
1. 先 `git add -A && git commit`（保存当前状态，这样还能找回）
2. 或者先 `git branch backup`（建个备份分支）

**更安全的替代方案**——用 `revert` 做"反向提交"：
```bash
git revert <提交号>      # 生成一个新提交，内容是"撤销这次改动"
```
好处：历史完整保留，可以再 revert 回来。

---

## 五、危险命令黑名单（执行前一定要三思）

| 命令 | 后果 |
|---|---|
| `git checkout .` | **丢弃所有未提交改动**（你的 bug 修复就没了） |
| `git restore .` | 同上 |
| `git reset --hard` | 丢弃未提交改动 + 回退提交指针 |
| `git clean -fd` | **删除所有未跟踪文件**（新建的文档、脚本全没） |
| `git push --force` | 覆盖远程历史（多人协作时是灾难） |

**防身三件套**（每次做危险操作前）：
```bash
git add -A && git commit -m "操作前的安全备份"    # 或
git branch backup-$(date +%m%d)                  # 建个备份分支
```

---

## 六、不想敲命令？用 VSCode 图形界面

VSCode 的 **Source Control 面板**（左侧栏第三个图标，或 `Ctrl+Shift+G`）能可视化做大部分操作：

| 想做什么 | 在面板里怎么点 |
|---|---|
| 看改了哪些文件 | 打开面板，直接列出来 |
| 看具体改动 | 点文件名 → 左右对比逐行显示 +/- |
| 提交 | 在消息框写说明 → 点 ✓ 提交 |
| 撤销某文件改动 | 右键文件 → "Discard Changes"（⚠️ 同上，会丢改动） |
| 查看文件历史 | 右键 → "Open Timeline"，能看到每次改动 |

**建议**：日常提交用 VSCode 面板（直观），回溯操作先看本文档再动手。

---

## 七、本项目当前建议操作

在把项目推到 GitHub 之前，建议**先做一次提交**，把这次的修复和文档都记录下来：

```bash
cd C:\Code\study\gohr_pytorch_reproduce
git add -A
git commit -m "fix: 修正输入表示，6轮 val_acc 0.60→0.78

- 原 view(16,4)+permute 按每4比特切分，破坏密码的面向字结构
- 改为 view(4,16)，等价于论文的 Reshape((4,16))+Permute((2,1))
- 新增：CSV/TensorBoard 记录、断点续训、可视化与启动脚本
- 文档：环境使用指南、VSCode说明、调试案例存档"
```

（提交信息写详细一点，以后回头看会感谢自己——这也是面试时能讲的故事。）

**注意**：`training_curves.png`、`train_6r_fixed.log` 这类训练产物，可以考虑加进 `.gitignore`（它们是产物，不是代码）。当前 `.gitignore` 已忽略 `checkpoints/`、`*.pt`，可以再补上：

```
# 训练产物
*.log
training_curves.png
```
