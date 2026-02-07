# Tmux 炼丹保活指南 (Remote Training Keep-Alive)

**适用场景**

- 跑动辄几小时/几天的深度学习训练
- 网络不稳定，SSH 经常自动断开
- 下班需要合上笔记本回家，但希望服务器上的训练继续跑

## 0. 安装 (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y tmux
```

## 1. 核心机制 (The "Why")

> **后台运行，前台直播**
>
> Tmux 将**程序运行**(Server)与**显示界面**(Client)解耦。即使断网、关机、退出 SSH，
> Server 端的训练进程依然在内存中运行，直到你下次回来重连画面。

## 2. 标准作业流程 (SOP)

### 第一步: 创建保险箱 (Start)

不要直接跑代码，先新建一个 Tmux 会话:

```bash
tmux new -s train_v1
# -s 后面是名字，建议以实验名称标记
```

### 第二步: 开始炼丹 (Run)

在 Tmux 窗口内正常运行代码。

**推荐用项目的标准入口**:

```bash
# 训练入口: 全流程 (由 training.yaml 控制具体阶段)
python scripts/full_train.py --config configs/training.yaml -- --dataset=ZELDA
```

**小规模调试用 dev 配置**:

```bash
python scripts/full_train.py --config configs/dev/dev_training.yaml
```

> 说明: `scripts/full_train.py` 会按 `configs/training.yaml` 中的开关
> 选择是否跑 video tokenizer / latent actions / dynamics。

### 第三步: 安全撤离 (Detach)

当你想离开时(或网络自动断了)，手动将任务挂起:

1. 按下 **Ctrl + b** (松开)
2. 按下 **d** (Detach)

结果: 你回到普通终端，提示 `[detached]`，但训练在后台继续跑。

### 第四步: 恢复现场 (Attach)

```bash
# 1. 查看有哪些会话在跑
tmux ls

# 2. 进入指定的会话
tmux attach -t train_v1
```

## 3. 常用指令速查表 (Cheat Sheet)

**核心前缀键 (Prefix)**: 所有 Tmux 快捷键都必须先按 **Ctrl + b**，松开后再按后续按键。

| 动作 | 命令/快捷键 | 说明 |
| --- | --- | --- |
| 新建会话 | `tmux` 或 `tmux new -s <名字>` | 开启一个新的工作区 |
| 挂起离开 | Prefix + `d` | 退出当前会话(程序不中断) |
| 查看列表 | `tmux ls` | 查看后台所有活着的会话 |
| 恢复会话 | `tmux attach -t <名字>` | 重新进入某个会话 |
| 杀死会话 | `tmux kill-session -t <名字>` | 彻底关闭某个会话(程序会停止) |
| 左右分屏 | Prefix + `%` | 一边看日志，一边看 `nvtop` |
| 上下分屏 | Prefix + `"` | 同上 |
| 切换面板 | Prefix + 方向键 | 在分屏之间光标跳转 |
| 关闭面板 | Ctrl + `d` (或输入 `exit`) | 关闭当前分屏 |

## 4. 炼丹师进阶技巧 (Pro Tips)

### 技巧 A: 查看之前的日志 (滚动模式)

**痛点**: 默认情况下鼠标滚轮无法在 Tmux 里向上翻页看 log。

**操作**:

1. 按 Prefix + `[` 进入 Copy Mode
2. 用方向键或 PageUp/PageDown 上下翻
3. 按 `q` 退出

### 技巧 B: 显卡监控分屏

建议把屏幕切成两半:

- 左边: 跑 `python scripts/full_train.py ...`
- 右边: 跑 `watch -n 1 nvidia-smi` 或 `nvtop`

### 技巧 C: 强制杀死卡死的会话

如果代码写了死循环把 Tmux 卡死了，进不去:

```bash
tmux kill-session -t train_v1
```

## 一句话总结

**进门 `tmux new`，出门 `Ctrl+b d`，回家 `tmux attach`。**
