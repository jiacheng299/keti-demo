# 开放场景视觉语义认知 Demo

本仓库用于构建项目答辩 Demo。当前开发状态、范围与验收口径见：

- [设计规范](docs/superpowers/specs/2026-09-16-open-scene-vision-demo-design.md)
- [实施计划](docs/superpowers/plans/2026-09-17-open-scene-demo-implementation.md)
- [功能说明](docs/03_feature_manual.md)
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

需要使用 DeepSeek API 时，复制 `.env.example` 为 `.env` 并在本机设置 `DEEPSEEK_API_KEY`。不要提交 `.env` 或把 API Key 写入代码。

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

## 测试

```powershell
python -m pytest -q
```

项目处于逐日开发阶段。README 与功能说明会随已验证能力更新。
