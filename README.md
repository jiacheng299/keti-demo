# 开放场景视觉语义认知 Demo

本仓库是一个可在 CPU 笔记本上运行的开放场景视频分析 Demo：通过离线模板或 DeepSeek 生成场景配置，再调用边防/火灾小模型和规则引擎生成可视化结果与事件证据。

- [设计规范](docs/superpowers/specs/2026-09-16-open-scene-vision-demo-design.md)
- [实施计划](docs/superpowers/plans/2026-09-17-open-scene-demo-implementation.md)
- [功能说明](docs/03_feature_manual.md)
- [详细用户手册](docs/04_user_guide.md)
- [测试与验收记录](docs/05_test_and_acceptance.md)
- [模型与样例选择记录](docs/model_selection.md)
- [14 天开发清单](worklog/daily_checklist.md)
- [每日工作记录](worklog/daily_log.md)

## 环境准备

建议使用 Python 3.11。PowerShell 中执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

需要使用 DeepSeek API 时，在启动应用的同一 PowerShell 窗口中执行：

```powershell
$env:DEEPSEEK_API_KEY="你自己的_API_Key"
```

项目不会自动读取 `.env`。不要把 API Key 写入代码、文档或 Git。

## 启动

这是 Streamlit Web 应用，不能使用 `python app.py` 或 IDE 的“运行 Python
文件”按钮直接启动。该方式只会出现 `missing ScriptRunContext` 警告，不会启动
网页服务。

请在项目根目录的 PowerShell 终端中执行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

终端显示 `Local URL: http://localhost:8501` 后，在浏览器打开该地址。停止服务时，
在终端按 `Ctrl+C`。

## 页面操作

1. 上传 MP4、AVI、MOV、OGV 或 MKV 视频。
2. 默认选择一个离线模板；如已配置 API Key，也可选择 DeepSeek 解析并填写场景需求。
3. 点击“生成场景配置”，检查页面显示的场景、模型、目标和规则。
4. 点击“开始分析”，等待进度达到 100%。
5. 播放结果视频，查看事件表，并按需下载视频、JSON、CSV、摘要和场景配置。

页面会在分析完成后将 OpenCV 输出转换为浏览器兼容的 H.264 视频。该步骤由
`imageio-ffmpeg` 提供的本地 FFmpeg 执行，不需要另外安装系统级 FFmpeg。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

两个固定场景各重复验收 3 次：

```powershell
Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\run_acceptance.py --repeat 3
```

页面操作、输出文件、故障处理和安全边界请阅读[详细用户手册](docs/04_user_guide.md)。
