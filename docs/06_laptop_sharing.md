# 用笔记本分享 Demo

Cloudflare 临时公网 HTTPS 隧道启动后，地址保存在本机 `runs/share/public-url.txt`。
局域网分享需按下方命令监听 `0.0.0.0`，其他设备访问 `http://<本机局域网IP>:8502/`。
使用 `ipconfig` 查看当前电脑的 IPv4 地址；换电脑或网络后不能沿用旧地址。

分享实例不再需要访问码。访客打开链接后可以直接上传视频、调用已配置的 DeepSeek API、
运行模型和下载自己的分析结果，但没有保存、清除本机 API 密钥的按钮。
拿到链接的人都可以使用 Demo；他们的 DeepSeek 调用会使用分享者的 API 额度。
历史 `runs/share/access-code.txt` 文件已不再读取，也不再生成访问码。

笔记本需要保持开机、联网且不休眠。视频分析在笔记本 CPU 上执行，建议先逐人测试短视频。
目前已验证本机通过局域网地址访问成功；其他设备的最终连通性还取决于路由器是否启用客户端隔离。

## 再次启动

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\serve_shared.py --host 0.0.0.0 --port 8502
```

前台运行时按 Ctrl+C 停止。后台运行时可在任务管理器中结束对应的分享服务 Python 进程。
不要重复启动已经占用 8502 端口的实例。

其他设备通过局域网访问时，Windows 防火墙需允许当前 Python 程序在可信网络接受连接。
新电脑不会继承开发者电脑上的防火墙规则；公网隧道为主动出站连接，无需开放路由器端口。

## 公网访问与重启

公网隧道连接到本机 8502 端口，不需要服务器、域名或路由器端口转发。
已通过公网 WSS 接口验证：未登录的新会话直接显示 Demo 和上传控件，没有访问码输入框，
API 密钥管理控件保持隐藏。
单个视频上限为 95 MB，以适应公网入口的请求大小限制；本地非分享实例仍为 500 MB。

在项目根目录使用 PowerShell 运行：

新电脑先安装官方 Windows amd64 版 cloudflared，将可执行文件保存为
`runs/share/bin/cloudflared.exe`（目录不存在时新建）。下载入口：
[Cloudflare 官方 GitHub Releases](https://github.com/cloudflare/cloudflared/releases)。
它不包含在 Git 仓库中，也不是本地运行必需依赖。

```powershell
.\scripts\start_public.ps1
```

脚本复用已运行的服务和隧道；如果未运行，就在后台启动。每次重建隧道会产生新地址，
以脚本输出及 `runs/share/public-url.txt` 为准。脚本不会设置开机自启。

目前未生成可用的短链接：尝试的短链服务出现人机验证、拒绝临时隧道域名或返回错误。
分享时使用上述文件中的完整 HTTPS 地址。即使之后创建了短链接，重建临时隧道后也需要
更新短链目标或重新生成短链；临时隧道本身不能自定义为简短固定域名。

结束公网分享：

```powershell
.\scripts\stop_public.ps1
```

此命令只停止该项目的隧道，保留本地 Demo 服务。断网或休眠期间无法访问。
临时隧道适合演示，不提供固定地址或持续可用性保证。
参见 [Cloudflare Quick Tunnels 官方说明](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。
