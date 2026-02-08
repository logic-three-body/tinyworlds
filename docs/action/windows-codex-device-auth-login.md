# Windows + VS Code Codex Device Code 登录复盘

## 失败原因
- CLI 包名错误：尝试安装 @codex-ai/cli，npm 404。
- PATH 未刷新：安装 Node.js 后当前 PowerShell 会话未生效，导致 node/npm/codex 不可用。
- 在 PowerShell 里直接用 "path" --version 语法报错，未使用 & 调用运算符。
- VS Code 扩展被强行指定 chatgpt.cliExecutable 指向手动路径（开发用途），导致扩展启动 Codex 进程时报 spawn EINVAL。
- Codex 交互式提示下进入了沙箱配置步骤，误把 whoami/hostname 等命令阻断当作登录失败，实际已登录成功。

## 成功经验
- 使用正确的包名安装 CLI：`npm i -g @openai/codex`。
- 本地会话补 PATH：`$env:Path += ";C:\Program Files\nodejs"`，或重开 PowerShell。
- 验证 CLI：`npx codex --version`，避免 PATH 未刷新问题。
- 设备码登录：`npx codex login --device-auth`，按提示完成浏览器授权。
- 删除 chatgpt.cliExecutable 自定义配置，避免扩展错误调用 CLI。
- 复查扩展日志定位原因：`Code - Insiders/logs/.../exthost/openai.chatgpt/Codex.log`。

## 关键结论
- CLI 登录成功后，扩展仍未登录通常是“扩展无法启动 Codex 进程”，而不是凭据问题。
- 正确做法是让扩展使用默认 CLI 解析路径，避免手动覆盖。
