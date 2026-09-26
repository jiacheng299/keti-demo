# Demo 用户使用手册

## 1. Demo 能做什么

本 Demo 在 Windows 笔记本上使用 CPU 处理本地视频。用户选择场景模板，或用 DeepSeek 把自然语言需求转成受白名单约束的 `SceneSpec`；系统再自动选择对应的本地小模型、跟踪器和规则，最终生成标注视频、事件、截图和结构化文件。

已接入的场景：

- 边防：人员进入禁区、车辆进入禁区、人员区域滞留。
- 火灾：火焰/烟雾目标检测，经连续帧规则从“疑似”升级为“确认”。当前固定素材只验证了火焰效果，不对烟雾效果作保证。

当前不包含摄像头/RTSP、多路并发、本地大模型、身份识别、模型训练和生产部署。

## 2. 环境和必需文件

- Windows 10/11。
- Python 3.11。
- 只使用 CPU，不需要 CUDA。
- 项目根目录：`C:\Users\陈宇鑫\Desktop\keti_demo`。
- 边防权重：`models/border/yolo11n.pt`。
- 火灾权重：`models/fire/fire_smoke_yolov8n.pt`。
- 建议验收视频：`assets/demo_videos/border_demo.avi` 和 `assets/demo_videos/fire_demo.avi`。

权重、视频、运行结果和密钥均已被 `.gitignore` 排除，不会推送到 GitHub。

## 3. 首次安装

在项目根目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证依赖状态：

```powershell
.\.venv\Scripts\python.exe -m pip check
```

如果 PowerShell 禁止执行 `Activate.ps1`，可不激活虚拟环境，后续一律使用 `.\.venv\Scripts\python.exe` 即可。

## 4. 配置 DeepSeek（可选）

离线模板完成所有核心流程，不需要 API。如需测试自然语言配置，在启动 Streamlit 的同一个 PowerShell 窗口中设置：

```powershell
$env:DEEPSEEK_API_KEY="你自己的_API_Key"
```

只对当前终端会话生效。项目不会自动读取 `.env`，也不应将密钥写入代码、文档或 Git。清除当前终端中的密钥：

```powershell
Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
```

## 5. 启动和停止

必须使用 Streamlit 启动，不能直接运行 `python app.py`：

```powershell
cd C:\Users\陈宇鑫\Desktop\keti_demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

终端出现 `Local URL: http://localhost:8501` 后，浏览器打开该地址。停止服务时，回到启动终端按 `Ctrl+C`。

## 6. 页面控件说明

### 6.1 运行状态

显示 DeepSeek API 是否已配置。这只影响“DeepSeek 解析”，不影响离线模板。模型只在分析期间加载，完成或失败后释放。

### 6.2 上传视频

支持 MP4、AVI、MOV、OGV 和 MKV，单文件不超过 500 MB。上传后显示分辨率、帧率、总帧数和时长。视频无法解码时会在模型加载前报错。

### 6.3 场景配置

`配置方式` 有两种：

- `离线模板`：最稳定，不访问网络。
- `DeepSeek 解析`：将需求解析为 JSON，随后仍由本地白名单校验；缺少 Key、请求失败或输出无效时，回退到当前选中的离线模板。

`离线模板` 有四项：边防人员禁区、边防车辆禁区、边防人员滞留、火灾检测。`DeepSeek 解析` 模式下需填写 `场景需求`。点击 `生成场景配置` 后，必须先检查页面显示的模型 ID、目标类别、规则和参数。

### 6.4 开始分析

只有在“视频有效”且“已生成场景配置”时按钮才可用。进度条显示已读取帧、已推理帧和已记录事件数。CPU 推理期间请不要刷新页面。页面暂无运行中停止按钮，因此建议使用 10–15 秒视频。

### 6.5 分析结果

页面显示读取帧数、推理帧数、检测数、事件数、模型和耗时；播放 H.264 结果视频；展示中文事件表和确认事件截图。视频画面只绘制检测框、类别和 Track ID，不显示置信度。

下载入口共五个：结果视频、事件 JSON、事件 CSV、运行摘要、场景配置。

## 7. 两个推荐流程

### 7.1 边防人员禁区

1. 上传 `assets/demo_videos/border_demo.avi`。
2. 选择 `离线模板` 和 `边防人员禁区`。
3. 点击 `生成场景配置`，确认 `model_id` 为 `yolo_general`。
4. 点击 `开始分析`。
5. 预期结果：画面出现人员检测框、类别和 ID，事件表出现“进入禁区”。

### 7.2 火灾连续帧确认

1. 上传 `assets/demo_videos/fire_demo.avi`。
2. 选择 `离线模板` 和 `火灾检测`。
3. 点击 `生成场景配置`，确认 `model_id` 为 `fire_smoke`。
4. 点击 `开始分析`。
5. 预期结果：事件表先出现“疑似火灾”，连续命中达标后出现“确认火灾”；确认事件有截图，疑似事件不保存截图。

## 8. 输出文件

每次分析写入独立目录 `runs/<时间>-<随机值>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 本次已校验的 SceneSpec |
| `events.json` | 完整事件字段，便于程序处理 |
| `events.csv` | 表格格式事件，UTF-8 BOM，可直接用 Excel 打开 |
| `summary.json` | 事件统计和运行指标 |
| `snapshots/*.jpg` | 确认事件证据图 |
| `result.mp4` | H.264/yuv420p 标注视频 |

`runs/` 整个目录已被 Git 忽略。删除该目录会永久删除本地分析结果，需先备份有用文件。

## 9. 自动测试和重复验收

运行全部自动测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

执行两个场景各 3 次的真实 CPU 验收：

```powershell
Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\run_acceptance.py --repeat 3
```

通过时会输出 6 次运行结果、`acceptance_report.json` 以及每个场景的最后一份备用结果。验收器会检查必需文件、JSON/CSV/摘要事件数一致性、截图可读性、视频帧数和 H.264 标识。

## 10. 常见问题

### 直接运行 `app.py` 出现 `missing ScriptRunContext`

原因是启动方式错误。使用 `.\.venv\Scripts\python.exe -m streamlit run app.py`。

### 按钮不可用

先上传能够解码的视频，再点击 `生成场景配置`。两个条件都满足后 `开始分析` 才可用。

### 提示权重不存在

核对第 2 节的两个固定路径。权重名和 `config/models.yaml` 必须一致。

### DeepSeek 显示未配置或解析失败

检查 Key 是否设置在启动 Streamlit 的同一终端中。不影响使用离线模板；网络/API 失败时系统会回退。

### CPU 分析太慢

使用已准备的 10–15 秒视频，关闭其他高负载程序，分析时不要重复点击或刷新。实测中边防 120 帧约 7.2–11.4 秒，火灾 275 帧约 14.0–14.8 秒；数值只代表当前电脑和固定视频。

### 浏览器不能播放结果

先确认页面已显示“分析完成”。系统在推理后还要做本地 H.264 转码。若转码报错，重启页面并重新分析；同时可从 `runs/backup/` 取用已验证的备用视频。

## 11. 已知限制和安全边界

- DeepSeek 只生成配置，不直接处理视频，其输出不会被当作代码执行。
- 小模型结果受素材、视角、光线和权重限制；演示结果不等于正式准确率、误报率或生产可用性证明。
- 火灾“疑似”是过程事件，因此无截图；“确认”事件保存截图。
- 边防事件需要 Track ID；火灾是画面级判断，Track ID 为空是正常行为。
- 页面没有运行中停止按钮；后端保留了停止契约，但本期 Web 未暴露。
- 不要将 API Key、私人视频或生成的 `runs/` 目录提交到 Git。
