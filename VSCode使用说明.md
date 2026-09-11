# 用 VSCode 运行训练（操作说明）

> 配置已就绪：`.vscode/settings.json`（解释器）、`launch.json`（运行配置）、`tasks.json`（任务）
> 环境已确认：Python 扩展、debugpy、Pylance 均已安装

---

## 一、首次打开（只做一次）

1. VSCode → **文件 → 打开文件夹** → 选择 `C:\Code\study\gohr_pytorch_reproduce`
2. 右下角会提示选择 Python 解释器（或按 `Ctrl+Shift+P` 输入 `Python: Select Interpreter`）
3. 选择：**`C:\Users\lmjkj\anaconda3\envs\pytorch-gpu\python.exe`**

   （`settings.json` 里已预设好，正常情况下会自动选中。选中后左下角状态栏会显示 `pytorch-gpu`）

✅ 完成。之后打开这个项目都无需再配。

---

## 二、三种运行方式（按推荐度排序）

### 方式 1：任务方式 ⭐ 长训练推荐

**`Ctrl+Shift+P` → 输入 `Run Task` → 选择要跑的任务**

可选任务：

| 任务 | 用途 |
|---|---|
| 训练：快速测试（3轮/2分钟） | 验证环境是否正常 |
| **训练：6 轮论文级** | 正式训练（推荐） |
| 训练：5 轮 / 7 轮论文级 | 其他轮数 |
| 训练：续训（--resume） | 从断点继续 |
| 工具：画训练曲线 | 生成 training_curves.png |
| 工具：启动 TensorBoard | 启动网页监控 |

**为什么推荐**：任务方式直接调用 Python 执行，**没有调试器开销**，速度最快；输出实时显示在终端面板。

### 方式 2：F5 / 绿色三角（调试面板）

点左侧栏的"运行和调试"图标（或按 `Ctrl+Shift+D`），顶部下拉选择配置，然后按 **F5**：

| 配置 | 用途 |
|---|---|
| ① 快速测试（3轮/10万样本，约2分钟） | 首次验证 |
| ②③④ 训练 5/6/7 轮（论文级） | 正式训练 |
| ⑤ 续训（从断点继续） | 断点恢复 |
| ⑥ 画训练曲线 | 出图 |

**适用场景**：需要在代码里打断点、单步调试的时候（比如你想看 `dataset.py` 里数据长什么样）。
**注意**：调试模式有性能损耗，**长训练不建议用这个**（可能慢 1.5-2 倍）。

### 方式 3：右键运行（最简单）

在 `train.py` 文件里右键 → **"在终端中运行 Python 文件"**（Run Python File in Terminal）

会用默认参数（5 轮 / 100 万样本 / 25 epoch）跑一遍。**缺点：参数固定，不能选轮数**——想改参数就用方式 1 或 2。

---

## 三、改参数怎么办

三种方式对应三种改法：

| 想改什么 | 怎么改 |
|---|---|
| 临时试一下 | 在 VSCode 终端里手敲命令（见下） |
| 长期改 | 编辑 `.vscode/launch.json` 或 `tasks.json` 里对应的 `args` 数组 |
| 加一个新配置 | 复制一份配置块，改 `name` 和 `args` |

**终端里手敲**（VSCode 内置终端，`Ctrl+`` 打开）：
```powershell
python train.py --rounds 6 --train-size 10000000 --epochs 200 --depth 10
python train.py --rounds 6 --train-size 10000000 --epochs 200 --depth 10 --resume
```

---

## 四、几个必须注意的点

1. **同一时间只能跑一个训练**——显存只有 8GB，跑第二个会报 `CUDA out of memory`。
   先确认没有别的训练在跑（终端面板是否有任务在运行）。

2. **终端关掉 = 训练停止**。任务面板右上角的垃圾桶图标（终止任务）或关闭终端，都会杀掉训练进程。
   不小心关了？用"续训"任务恢复（`--resume`）。

3. **笔记本别休眠、别合盖**（见项目里的 `设置不休眠-右键管理员运行.bat`）。

4. **中文乱码问题**已解决：`settings.json` 里设了 `files.encoding: utf8`，配置里也传了 `PYTHONIOENCODING=utf-8`。

5. VSCode 的 **Source Control 面板**能直接看到 git 状态（左侧栏第三个图标），改完代码可以直接在里面提交。

---

## 五、日常流程建议

```
① 打开 VSCode（项目已记忆在上次打开的文件夹）
② 看训练进度：终端面板，或双击 checkpoints/history_6r.csv
③ 想跑新训练：Ctrl+Shift+P → Run Task → 选对应任务
④ 想看曲线：任务 → "工具：画训练曲线" → 打开生成的 png
⑤ 想实时看曲线：任务 → "工具：启动 TensorBoard" → 浏览器开 localhost:6006
⑥ 改完代码想提交：左侧 Source Control 面板 → 写 commit message → 提交
```

---

## 附：三种方式的对比总结

| 方式 | 速度 | 能改参数 | 能打断点 | 适合 |
|---|---|---|---|---|
| 任务（Ctrl+Shift+P → Run Task） | ⚡ 最快 | ✅ | ❌ | **长训练** |
| 调试（F5） | 🐢 较慢 | ✅ | ✅ | 调试代码 |
| 右键运行 | ⚡ 快 | ❌ | ❌ | 快速看一眼 |
