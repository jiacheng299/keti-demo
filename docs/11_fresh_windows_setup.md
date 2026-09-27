# 从 GitHub 下载后在新 Windows 电脑运行

目标：Windows 10/11、x64 处理器和 Python 3.13 64 位。使用 CPU，不需要 NVIDIA 显卡或 CUDA。当前不对 Windows ARM、32 位 Python、macOS/Linux 或其他 Python 版本作已验证承诺。

## 最短流程

1. 从 [Python 官方 Windows 下载页](https://www.python.org/downloads/windows/) 安装 Python **3.13.x 的 Windows installer (64-bit)**，勾选加入 PATH。安装后重新打开终端。
2. 登录有权限访问本项目的 GitHub 账号，克隆仓库，或选择 **Code → Download ZIP** 并完整解压。私有仓库需要仓库访问权限。不要在 ZIP 预览窗口内直接启动。
3. 将项目放在自己有读写权限的目录，例如 `D:\Projects\keti-demo`；中文和空格路径可用，不要放入需要管理员写权限的 Program Files。
4. 双击 **setup.cmd**，等待显示 **Setup completed**。安装失败会停止并保留错误提示，不会把未完成的安装当作成功。
5. 双击 **start.cmd**，打开 **http://127.0.0.1:8501**。保留启动窗口；按 Ctrl+C 可停止网页。
6. 上传 `assets/demo_videos` 中的视频，选择离线模板，生成并应用配置，开始分析。

如果终端里能使用 Python 3.13，但没有 `py` 命令，`setup.cmd` 会自动尝试 `python`。也可以运行 `python scripts/setup_demo.py`。安装器会检查版本和位数，不会向全局 Python 安装项目依赖。

## 首次安装自动完成的工作

- 新建项目内 `.venv`，不读取系统 site-packages，也不需要手动激活环境。
- 从 PyTorch 官方 CPU 源安装固定版 PyTorch/TorchVision，不安装 CUDA 工具链。
- 根据 `requirements.txt` 和 `requirements-windows.lock.txt` 安装验证过的依赖，执行 `pip check`。
- 按 `config/model_downloads.json` 下载 YOLO 通用检测、火焰/烟雾和人脸关键点三套权重，总计约 15.6 MB。下载地址固定版本，校验大小和 SHA-256；网络中断或校验失败时不覆盖现有权重，重新执行即可重试。
- 检查示例视频读取、三套模型 CPU 推理、双人关键点检测和浏览器 H.264 转码。结果写入 `runs/install-check/report.json`，不调用 DeepSeek/Qwen，也不消耗 API 额度。

首次安装依赖下载量明显大于模型文件，需要预留数 GB 磁盘空间及联网时间。安装完成且权重齐全后，四种离线模板可以断网运行。

## API 与公网

API Key 不包含在 GitHub 仓库中，也不会自动从旧电脑迁移。离线功能不需要 Key。要使用云端功能，在新电脑的本机网页“运行状态”分别保存 DeepSeek/Qwen Key，并设置有权限使用的 Qwen 模型和百炼地域地址。

本地网页与公网分享是两件独立的事：`start.cmd` 默认仅启动本机 8501 页面。新电脑不会沿用旧电脑的公网地址。需要分享时，按 [笔记本分享说明](06_laptop_sharing.md) 安装官方 cloudflared 并运行对应脚本；公网地址在隧道启动后生成。

## 排查与手动命令

| 现象 | 处理方式 |
| --- | --- |
| 提示找不到 Python、版本不符 | 安装 Python 3.13 x64；重新打开终端，用 `py -3.13 --version` 检查。已有 3.11/3.14 可以并存，安装器需要 3.13 |
| 无法访问 PyPI、PyTorch、GitHub、Hugging Face 或 Google Storage | 检查网络和本人可用代理后重跑 setup.cmd。安装器读取系统已有代理，不写死旧电脑的代理端口，也不关闭 TLS 校验 |
| 已有 `.venv` 不能使用或来自其他电脑 | 将旧 `.venv` 改名备份，再运行 setup.cmd 重建；虚拟环境不能直接跨电脑搬运 |
| `DLL load failed`、缺少 `VCRUNTIME140` 等 | 从 [微软官方页面](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) 安装 Visual C++ 2015–2022 x64 运行库，重启终端后重试 |
| 权重下载失败、文件不完整 | 重跑 `scripts/download_models.py`；它只使用校验通过的完整文件。下载清单位于 `config/model_downloads.json` |
| 8501 端口被占用 | CMD 执行 `netstat -ano \| findstr :8501` 查看占用；确认已有网页，或用下面的手动命令选择其他端口 |
| Key 保存失败 | 检查 Windows 凭据管理器，也可在启动窗口设置环境变量 `DEEPSEEK_API_KEY`、`QWEN_API_KEY`；不会影响离线分析 |

已安装依赖后，单独重新下载或检查模型：

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/download_models.py --check
.\.venv\Scripts\python.exe scripts/check_installation.py
```

手动安装依赖（从项目根目录执行）：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-windows.lock.txt
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/check_installation.py
```

手动换端口启动（例如 8503）：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8503
```

模型路径已根据项目配置文件的位置解析，不依赖旧电脑的用户名或绝对目录；启动脚本会自动进入自身所在目录。

## 验证范围

2026-09-27 已在当前 Windows 电脑的独立目录完成隔离安装验证：从 Git 归档与待交付源码建立干净副本，没有复制原项目的 `.venv`、权重、运行目录或设置；重新创建 Python 3.13.5 环境，`include-system-site-packages=false`。依赖通过 pip 安装（可复用下载缓存），三套权重均从清单指定地址实际下载并校验。

- `setup_demo.py` 从空环境执行成功，随后实际运行 `setup.cmd` 验证重复安装可用。
- `pip check` 通过，安装版本符合依赖约束，PyTorch 为 `2.14.0+cpu`。
- 干净副本及其虚拟环境中，**237 项测试通过**。
- 使用独立端口启动真实 Streamlit 服务，健康检查和 WebSocket 页面均正常；使用空凭据后，网页显示两家 API 未配置，仍能生成离线场景配置。没有调用云端 API。
- 四个场景使用下载的真实权重完整运行，结果视频均已转码并逐帧重新解码，事件截图与 JSON/CSV/规则过程文件均存在：

| 场景 | 视频帧数 | 事件数 |
| --- | ---: | ---: |
| 边防禁区 | 120 | 3 |
| 火灾 | 275 | 2 |
| 人员在岗 | 2050 | 2 |
| 多人闭眼（双路拼接片） | 203 | 4（两个不同 ID 分别报警、恢复） |

本机验收报告保留在 `runs/portability-check/clean checkout/runs/install-check/`，不提交运行产物。验证的是新目录和新依赖环境，**不是另一台实体电脑**；系统 DLL、处理器架构和网络可达性仍以新电脑实际自检结果为准。
