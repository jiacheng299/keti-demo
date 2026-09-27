# 开放场景视觉语义认知 Demo

本仓库是一个可在 CPU 笔记本上运行的开放场景视频分析 Demo：通过离线模板或 DeepSeek 生成场景配置，再调用边防/火灾小模型和规则引擎生成可视化结果与事件证据。

离线模板还支持人员在岗监测（岗位人数、持续缺员及回岗）和打瞌睡监测（最多 5 张清晰人脸分别计算持续闭眼及睁眼恢复）。多人测试视频与验证结果见 [多人眼部监测](docs/10_multi_face_drowsiness.md)。
前者复用 YOLO11n，后者使用 MediaPipe Face Landmarker，全部在 CPU 上运行。
这两个新增场景也支持 DeepSeek 解析；解析前先检查输入是否为明确的监测需求、现有模型与规则能否完整执行。
闲聊、模糊或超出能力的需求会显示原因并停止生成配置，不会自动套用模板。
详见 [需求校验与多模态接入建议](docs/08_requirement_validation_and_vlm_plan.md)。
详细设计和实际验收见 [在岗与打瞌睡监测](docs/07_duty_and_drowsiness_design.md)。

- [项目目录树与文件作用](docs/00_project_structure.md)
- [设计规范](docs/superpowers/specs/2026-09-16-open-scene-vision-demo-design.md)
- [实施计划](docs/superpowers/plans/2026-09-17-open-scene-demo-implementation.md)
- [功能说明](docs/03_feature_manual.md)
- [详细用户手册](docs/04_user_guide.md)
- [测试与验收记录](docs/05_test_and_acceptance.md)
- [模型与样例选择记录](docs/model_selection.md)
- [14 天开发清单](worklog/daily_checklist.md)
- [每日工作记录](worklog/daily_log.md)

## 环境准备

仓库包含 `assets/demo_videos/` 中的示例视频、素材来源和许可说明，克隆后即可选取视频测试。
模型权重仍需按 `config/models.yaml` 与 [模型选择记录](docs/model_selection.md) 准备；
人脸权重可运行下方下载命令获取。API Key、本机上传记录与 `runs/` 分析结果不随仓库分发。

建议使用 Python 3.11。PowerShell 中执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_face_model.py
```

当前 Windows 笔记本已在 Python 3.13 上验证 MediaPipe 1.0.1。OpenCV 两个发行包分别满足
Ultralytics 与 MediaPipe 的依赖要求，版本固定一致；请一起按 requirements.txt 安装，避免单独升级其中一个。

可以在本机网页“运行状态”中分别展开“DeepSeek API 设置”和“Qwen API 设置”，
在密码框输入自己的 API Key，然后点击“保存 / 修改 API Key”。两家的密钥独立保存，输入新 Key 再保存即可替换旧 Key。密钥保存在本机系统
凭据管理器，重启后仍可使用；“清除已保存 Key”可以删除它。保存本身不调用
接口，只有选择“DeepSeek 解析”并点击“生成场景配置”时才发送场景需求。Qwen 已接入云端抽帧检验：本地不支持且适合视觉判断的需求，会显示“使用 Qwen 检验”入口；点击后才上传采样画面。输入框不会回填已保存的密钥；公网分享页面隐藏编辑入口。

Qwen 默认使用 `qwen3-vl-32b-thinking` 和北京官方 Base URL。模型、地址、每日调用额度和公网开关可在本机“Qwen API 设置”中保存，也可点击“测试 Qwen 连接”。上传自己的视频，填写需求后生成配置并启动 Qwen 检验。完整操作见 [Qwen 视频检验](docs/09_qwen_video_inspection.md)。

本机管理页面可使用 `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501` 启动，访问 `http://127.0.0.1:8501`。它与8502端口的分享服务使用同一系统凭据，保存后刷新分享页面即可更新状态。

也可以在启动应用的同一 PowerShell 窗口中配置环境变量，作为备用入口：

```powershell
$env:DEEPSEEK_API_KEY="你自己的_API_Key"
```

项目不会自动读取 `.env`。不要把 API Key 写入代码、文档或 Git。
已保存的系统凭据优先于环境变量；删除保存的 Key 不会清除环境变量。
页面会显示配置来源。接口鉴权失败、余额不足、网络异常或返回配置无效时，
会明确提示原因，不生成配置；可以重试，或手动切换到“离线模板”选择场景。
场景解析显式关闭思考模式，并为最终 JSON 预留 4096 tokens；请求单次超时为
60 秒，传输失败最多重试一次。空响应、输出截断、超时和连接错误会分别提示。
使用“火焰持续8秒”等时间需求时，先上传视频；系统将实际帧率提供给解析器，
页面显示确认帧数及对应持续时间。25 FPS 视频从首次命中到满8秒需201个命中帧。
DeepSeek 提取持续秒数，由程序按实际帧率换算命中帧数。更换不同帧率的视频后，
在配置卡片中检查新阈值并点击“应用配置”，无需重新请求 API。

## 启动

这是 Streamlit Web 应用，不能使用 `python app.py` 或 IDE 的“运行 Python
文件”按钮直接启动。该方式只会出现 `missing ScriptRunContext` 警告，不会启动
网页服务。

模型运行时默认将 Ultralytics 配置保存在项目内的 `runs/ultralytics/`，
避免后台运行时写入 Windows 用户目录出现权限错误。如已设置
`YOLO_CONFIG_DIR`，则使用该环境变量指定的位置。

请在项目根目录的 PowerShell 终端中执行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

终端显示 `Local URL: http://localhost:8501` 后，在浏览器打开该地址。停止服务时，
在终端按 `Ctrl+C`。

## 页面操作

1. 上传 MP4、AVI、MOV、OGV 或 MKV 视频。
2. 选择“离线模板”时只显示模板选择；选择“DeepSeek 解析”时只显示场景需求。切换模式或修改输入后，需要重新生成配置。
3. 点击“生成场景配置”，通过配置卡片检查或修改目标、规则阈值、漏检容忍和冷却时间。边防场景可在视频参考帧上绘制矩形/多边形、拖动顶点、命名区域及添加多个区域规则，修改后点击“应用配置”。
4. 点击“开始分析”，等待进度达到 100%。
5. 播放结果视频，查看事件表，并按需下载视频、JSON、CSV、摘要和场景配置。

禁区统一使用“边防禁区”模板，监测目标仅分“人员”和“车辆”。小汽车、摩托车、
客车、卡车的模型结果统一映射为 `car`，再参与跟踪、规则判断和结果导出。

页面会在分析完成后将 OpenCV 输出转换为浏览器兼容的 H.264 视频。该步骤由
`imageio-ffmpeg` 提供的本地 FFmpeg 执行，不需要另外安装系统级 FFmpeg。

结果视频会绘制场景配置中的禁区：平时为黄色边界及淡色填充，触发事件的区域和
相关目标框短暂高亮为红色。视频顶部显示中文事件名称、目标/跟踪 ID 和触发时间，
按视频时间保留 3 秒；疑似火灾为橙色，确认火灾为红色。事件截图保留相同标注。
中文字体不可用的系统自动使用英文标签。网页播放器居中显示，最大宽度 640 像素。

视频底部显示实际规则进度，例如“滞留 3.6/5 秒”“火灾确认 6.2/8 秒”，并说明
未报警原因。结果页“规则过程与未报警原因”可按视频时间查看全部规则状态；
`rule_progress.jsonl` 保留每个推理帧的完整进度，可从网页下载。绘图坐标使用原视频
像素，不随网页尺寸改变；修改配置后需重新分析视频。

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
