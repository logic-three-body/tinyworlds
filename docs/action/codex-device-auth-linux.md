# WSL 环境迁移操作手册（C 盘 -> D 盘）

这份手册按“定位现状 -> 决策确认 -> 执行迁移 -> 恢复配置”的顺序组织，目标是在保证数据安全的前提下把 WSL 发行版从 C 盘迁移到 D 盘。

> 前置条件：以下命令请在 `PowerShell（管理员）` 中按顺序执行。

---

## 第一阶段：定位现状（Check）

先确认当前发行版名称和实际占用位置。

### 1. 查看发行版名称

```powershell
wsl -l -v
```

记下你要迁移的发行版名称（例如：`Ubuntu` 或 `Ubuntu-20.04`）。

### 2. 定位当前 `ext4.vhdx` 文件

```powershell
Get-ChildItem -Path $env:LOCALAPPDATA\Packages -Filter "ext4.vhdx" -Recurse -ErrorAction SilentlyContinue |
  Select-Object FullName, Length
```

判断依据：
- 如果路径以 `C:\` 开头，说明该发行版当前主要占用 C 盘。
- `Length` 越大，占用空间越大（通常是几十 GB）。

---

## 第二阶段：决策确认（Decision）

执行前请确认：

1. 源位置：WSL 当前确实位于 C 盘。
2. 目标位置：D 盘有足够空间。
- 经验值：目标盘可用空间 >= 当前 `ext4.vhdx` 的 1.2 倍（用于暂存备份包 + 新安装目录）。
3. 风险控制：如果有不可替代的重要代码，先额外复制一份到 Windows 目录做双重备份。

确认后再进入第三阶段。

---

## 第三阶段：执行迁移（Execution）

示例参数：
- 发行版名称：`Ubuntu`
- 目标路径：`D:\WSL\Ubuntu`
- 备份包：`D:\ubuntu-backup.tar`

请按你的实际名称和路径替换命令。

### 1. 停止 WSL

```powershell
wsl --shutdown
```

### 2. 导出镜像（备份）

```powershell
# 格式: wsl --export <发行版名称> <备份文件路径>
# 该步骤可能较慢，等待命令完成后再继续
wsl --export Ubuntu D:\ubuntu-backup.tar
```

可选校验（建议执行）：

```powershell
Get-Item D:\ubuntu-backup.tar | Select-Object FullName, Length, LastWriteTime
```

### 3. 注销原系统（删除 C 盘安装）

高风险步骤：执行前务必确认 `D:\ubuntu-backup.tar` 已成功生成且大小正常。

```powershell
wsl --unregister Ubuntu
```

### 4. 导入到新盘

```powershell
# 1) 创建目标目录
New-Item -Path "D:\WSL\Ubuntu" -ItemType Directory -Force

# 2) 导入发行版
# 格式: wsl --import <新名称> <安装路径> <备份包路径>
wsl --import Ubuntu D:\WSL\Ubuntu D:\ubuntu-backup.tar
```

导入完成后，可验证：

```powershell
wsl -l -v
```

如果版本不是 WSL2，可切换：

```powershell
wsl --set-version Ubuntu 2
```

---

## 第四阶段：恢复配置（Configuration）

导入后的发行版通常默认以 `root` 登录，需要改回常用用户。

### 1. 启动发行版

```powershell
wsl -d Ubuntu
```

### 2. 在 WSL 内设置默认用户

```bash
nano /etc/wsl.conf
```

写入：

```ini
[user]
default=你的用户名
```

保存并退出后，回到 PowerShell 执行：

```powershell
wsl --terminate Ubuntu
wsl -d Ubuntu
```

---

## 第五阶段：收尾（Cleanup）

1. 验证环境：检查代码、`conda`、Python 依赖、SSH key 等是否完整。
2. 空间回收：确认全部正常后删除备份包。

```powershell
Remove-Item D:\ubuntu-backup.tar
```

