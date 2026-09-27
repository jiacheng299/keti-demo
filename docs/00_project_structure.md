# 项目目录树与文件作用

更新日期：2026-09-27。当前包含边防禁区、火焰/烟雾、人员在岗、多人持续闭眼监测，以及 DeepSeek 需求解析和 Qwen 视频抽帧检验。

下面列出项目维护的源码、配置、脚本、测试和文档。各 Python 目录中的 `__init__.py` 用于声明包及组织对外导入；缓存、虚拟环境和每次分析产生的文件不逐项展开。

## 入口、配置与素材

```text
keti-demo/
├── app.py                              # Streamlit 网页入口，组织上传、配置、分析与结果展示
├── setup.cmd                           # Windows 首次安装入口，选择 Python 3.13 并执行安装器
├── start.cmd                           # Windows 本机启动入口，先检查权重再启动 8501 页面
├── README.md                           # 项目介绍、安装、启动与使用入口
├── requirements.txt                    # Python 依赖及版本约束
├── requirements-windows.lock.txt       # Windows x64 / Python 3.13 验证过的完整依赖约束
├── pytest.ini                          # 测试发现配置，限定从 tests 目录收集测试
├── .env.example                        # 环境变量填写示例；应用不会自动读取 .env
├── .gitignore                          # 排除密钥、虚拟环境、权重及运行产物；允许提交示例视频
├── config/
│   ├── models.yaml                     # 模型注册配置：适配器、权重、类别、置信度、CPU 参数
│   ├── model_downloads.json            # 三套模型的固定下载地址、文件大小和 SHA-256
│   ├── settings.yaml                   # 预留的全局默认配置；页面行为以 app.py 当前实现为准
│   └── scenes/
│       ├── border_intrusion.yaml       # 网页统一使用的边防禁区模板
│       ├── border_person_intrusion.yaml# 保留的人员禁区模板，兼容旧配置
│       ├── border_vehicle_intrusion.yaml# 保留的车辆禁区模板，车辆统一使用 car
│       ├── border_dwell.yaml           # 区域内持续滞留模板
│       ├── fire_detection.yaml         # 火焰/烟雾连续命中确认模板
│       ├── on_duty.yaml                # 岗位最低人数、缺员时长和回岗恢复模板
│       └── drowsiness.yaml             # 多人独立闭眼计时，共享 EAR、时长和画面质量阈值
├── assets/demo_videos/
│   ├── SOURCES.md                      # 示例视频来源、许可和处理方式（提交 Git）
│   ├── monitoring_samples.json         # 示例尺寸、帧率、时长、来源和 SHA-256（提交 Git）
│   ├── OpenVino-Driver-Behaviour-LICENSE.txt # 对应示例素材的 Apache-2.0 许可（提交 Git）
│   ├── border_demo.avi                 # 边防检测演示片
│   ├── fire_demo.avi                   # 火焰检测演示片
│   ├── vtest.avi                       # 制作边防示例使用的 OpenCV 原始素材
│   ├── fire_burning.ogv                # 制作火焰示例使用的原始素材
│   ├── on_duty_demo.mp4                # 岗位缺员与回岗示例
│   ├── drowsiness_demo.mp4             # 单人持续闭眼与恢复示例
│   ├── head_turn_control.mp4           # 转头对照及 Qwen 检验示例
│   ├── multi_person_clear_1080p.mp4     # 真实双人清晰眼部视频
│   └── multi_face_staggered_composite.mp4 # 同一演员双路错时拼接，验证独立计时
└── models/                             # 下载后的模型权重，均不提交 Git
    ├── border/yolo11n.pt               # YOLO 通用检测，识别人和车辆，也用于岗位人数统计
    ├── fire/fire_smoke_yolov8n.pt       # 火焰、烟雾检测模型
    └── face/face_landmarker.task        # MediaPipe 人脸与眼部关键点模型
```

## 核心源码

```text
src/
├── __init__.py                         # 源码根包
├── llm/                                # 自然语言解析与云端模型接口
│   ├── __init__.py                     # 包定义
│   ├── deepseek_client.py              # DeepSeek JSON 请求、超时、重试与异常提示
│   ├── prompts.py                      # 场景配置提示词、模型能力和 JSON 约束
│   ├── requirement_guard.py            # 检查需求是否明确、能否完整执行、是否可交给 Qwen
│   ├── scene_parser.py                 # 加载离线模板，解析并校验 DeepSeek 场景配置
│   └── qwen_client.py                  # Qwen 多模态请求、模型/地址设置和响应处理
├── models/                             # 将不同小模型适配为统一接口
│   ├── __init__.py                     # 包定义
│   ├── base_adapter.py                 # 模型 load/predict/unload 接口约定
│   ├── model_registry.py               # 读取可信模型配置，按白名单创建适配器
│   ├── yolo_adapter.py                 # YOLO 推理，归一化检测框，车辆类别统一映射为 car
│   ├── fire_adapter.py                 # 火焰/烟雾模型适配，复用 YOLO 推理
│   └── face_landmarker_adapter.py      # 最多 5 张人脸关键点、双眼 EAR 和侧脸角度测量
├── tracking/                           # 跨帧目标编号
│   ├── __init__.py                     # 包定义
│   ├── tracker.py                      # 根据类别和中心距离跟踪人员/车辆
│   └── face_tracker.py                 # 相邻帧人脸框匹配；消失或匹配不明确时重新编号
├── rules/                              # 根据检测结果与视频时间判定事件
│   ├── __init__.py                     # 包定义
│   ├── base_rule.py                    # 视频帧状态、候选事件与规则接口
│   ├── rule_engine.py                  # 按配置创建规则、汇总事件、处理报警开关与冷却
│   ├── region_rules.py                 # 多边形内外判断，以及进入、离开事件
│   ├── temporal_rules.py               # 按目标 ID 累计区域滞留时间
│   ├── fire_rules.py                   # 连续火焰/烟雾检测的疑似和确认状态
│   ├── occupancy_rules.py              # 区域人数、持续缺员、启动宽限与回岗恢复
│   ├── drowsiness_rules.py             # 每张脸独立判断连续闭眼、报警与稳定睁眼恢复
│   └── progress.py                     # 可显示、可导出的规则进度和未报警原因
├── schemas/                            # Pydantic 数据结构与校验
│   ├── __init__.py                     # 包定义
│   ├── detection.py                    # 标准检测结果：类别、框、置信度和跟踪 ID
│   ├── face_observation.py             # 人脸 ID、眼部关键点、EAR、角度与观测有效性
│   ├── event.py                        # 正式事件记录，包含截图位置和视频时间
│   ├── scene_spec.py                   # 场景、目标、规则和报警配置的数据结构
│   └── scene_time.py                   # 根据实际视频帧率解析时长规则，换算必要帧数
├── pipeline/                           # 串联完整分析流程
│   ├── __init__.py                     # 包定义
│   ├── analysis_pipeline.py            # 读视频 → 小模型 → 跟踪 → 规则 → 标注及导出
│   ├── contracts.py                    # 分析进度、运行摘要和结果路径的接口结构
│   └── qwen_pipeline.py                # 视频抽帧分窗、Qwen 检验、证据核验、缓存及结果导出
├── video/                              # 视频读写与浏览器适配
│   ├── __init__.py                     # 包定义
│   ├── video_source.py                 # 检查并读取视频，提供帧率、尺寸和逐帧时间戳
│   ├── frame_sampler.py                # 决定哪些视频帧进入模型推理
│   ├── video_writer.py                 # 保存带标注的结果视频
│   └── browser_video.py                # 使用 FFmpeg 转为浏览器可播放的 H.264
├── visualization/                      # 在视频像素上绘制结果
│   ├── __init__.py                     # 包定义
│   ├── annotator.py                    # 绘制检测框、类别及目标 ID
│   └── scene_overlay.py                # 绘制禁区、眼部关键点、每人状态、事件提示及计时
├── events/                             # 事件证据持久化
│   ├── __init__.py                     # 包定义
│   ├── event_manager.py                # 事件编号、重复过滤和截图保存
│   └── event_exporter.py               # 导出 JSON、CSV、配置、统计和摘要
├── ui/                                 # 网页组件及 UI 辅助逻辑
│   ├── __init__.py                     # 包定义
│   ├── app_service.py                 # 保存上传文件、解析配置、创建运行目录和整理展示数据
│   ├── scene_editor.py                # 场景参数卡片、区域规则编辑及配置校验
│   ├── region_canvas/index.html       # 浏览器矩形/多边形绘图、顶点拖动交互
│   ├── api_settings.py                # DeepSeek/Qwen Key 保存修改、模型选项与连接测试界面
│   ├── share_access.py                # 识别分享实例，限制分享页面修改本机凭据
│   ├── progress_view.py               # 按视频时刻查看所有规则状态与未报警原因
│   └── qwen_view.py                   # Qwen 手动检验入口、进度和结果界面
├── security/                           # 本机凭据和云端调用控制
│   ├── __init__.py                     # 包定义
│   ├── credential_store.py            # 使用操作系统凭据管理器保存、读取和删除 API Key
│   └── qwen_budget.py                 # SQLite 跨进程调用额度统计及并发占用控制
└── validation/
    ├── __init__.py                     # 包定义
    └── acceptance.py                  # 重复运行固定场景，核验视频、事件、截图和导出文件
```

## 启动、下载与验证脚本

```text
scripts/
├── setup_demo.py                       # 创建虚拟环境、安装 CPU 依赖、下载权重并运行自检
├── download_models.py                 # 三套模型下载与校验，原子替换、失败清理、支持仅检查
├── check_installation.py               # 无 API 检查真实模型推理、双人检测、视频读取和转码
├── serve_shared.py                     # 启动分享版网页；隐藏密钥编辑，按模式设置上传上限
├── start_public.ps1                    # 启动/复用分享服务与 Cloudflare 隧道，输出公网地址
├── stop_public.ps1                     # 停止本项目的公网隧道，本地网页可继续使用
├── download_face_model.py              # 下载官方 Face Landmarker 权重并校验摘要
├── prepare_multiface_samples.py        # 下载双人原片、制作错时拼接片、完整解码并记录元数据
├── validate_border_model.py            # CPU 运行边防检测样例并记录标准化检测结果
├── validate_fire_model.py              # 使用正例和负例视频验证火焰/烟雾模型
├── run_acceptance.py                   # 执行边防、火灾真实模型验收及重复运行
└── verify_qwen_demo.py                 # 实际调用 DeepSeek 与 Qwen 验证转头需求流程，会消耗额度
```

## 自动测试

```text
tests/
├── conftest.py                         # 测试公共夹具，隔离凭据、Qwen 设置和额度数据库
├── test_acceptance.py                  # 验收流程与产物校验
├── test_annotator.py                   # 检测框绘制与原图不被修改
├── test_api_settings.py                # Key 设置、保存修改和分享页可见性
├── test_app_smoke.py                   # 网页主要流程与结果展示冒烟测试
├── test_browser_video.py               # 浏览器兼容转码及错误处理
├── test_credential_store.py            # 系统凭据的读取、保存、删除与失败处理
├── test_deepseek_client.py             # DeepSeek 请求、响应、重试和异常处理
├── test_event_exporter.py              # JSON、CSV、摘要及配置导出
├── test_event_manager.py               # 事件去重、编号及截图
├── test_fire_adapter.py                # 火焰/烟雾模型适配
├── test_fire_model_validation.py        # 火灾模型验证脚本
├── test_fire_rules.py                   # 火灾疑似、确认与连续命中规则
├── test_frame_sampler.py               # 视频帧采样策略
├── test_model_output.py                # 统一检测结果的数据校验
├── test_model_registry.py              # 模型注册、配置校验及白名单
├── test_monitoring_rules.py            # 岗位缺员/回岗和眼部闭合/恢复时序
├── test_monitoring_ui.py               # 在岗、打瞌睡模板的参数编辑与应用
├── test_multi_face_drowsiness.py       # 多脸编号、独立计时/恢复、消失与交叉等情况
├── test_pipeline_smoke.py              # 模型、规则、视频和事件导出的串联流程
├── test_portable_setup.py              # 模型下载校验、网络重试、失败保护与独立于工作目录的路径
├── test_qwen.py                        # Qwen 设置、抽帧、接口、证据、缓存及额度控制
├── test_region_rules.py                # 禁区进入/离开及边界判断
├── test_requirement_guard.py           # 需求合法性、能力匹配和多模态路径判定
├── test_rule_progress.py               # 规则进度与解释记录
├── test_scene_editor.py                # 区域几何、参数编辑和应用配置
├── test_scene_overlay.py               # 禁区与事件在视频中的叠加显示
├── test_scene_parser.py                # 模板加载与模型输出的配置校验
├── test_scene_spec.py                  # 场景数据结构与约束
├── test_share_access.py                # 本机/分享模式识别与访问行为
├── test_temporal_rules.py              # 持续滞留等时间规则
├── test_tracker.py                     # 人员/车辆跨帧跟踪
├── test_ui_service.py                  # 上传、配置、结果表格和路径辅助逻辑
├── test_video_source.py                # 视频输入和元信息检查
└── test_yolo_adapter.py                # YOLO 推理接口、框归一化与车辆类别合并
```

## 文档与工作记录

```text
docs/
├── 00_project_structure.md             # 本目录树和文件作用说明
├── 03_feature_manual.md                # 功能与模块说明
├── 04_user_guide.md                    # 网页操作、参数及故障处理手册
├── 05_test_and_acceptance.md           # 测试和验收说明
├── 06_laptop_sharing.md                # 笔记本本地、局域网和公网分享方式
├── 07_duty_and_drowsiness_design.md     # 在岗与闭眼检测的模型、规则和实现依据
├── 08_requirement_validation_and_vlm_plan.md # 需求校验与多模态补充方案
├── 09_qwen_video_inspection.md         # Qwen 设置、测试流程和结果解释
├── 10_multi_face_drowsiness.md         # 多人眼部跟踪、视频使用与实测结果
├── 11_fresh_windows_setup.md           # 新 Windows 电脑一键安装、运行、API 配置及故障处理
├── model_selection.md                 # 模型和示例素材的选择依据
└── superpowers/                        # 历史设计记录；当前行为以源码和用户手册为准
    └── specs/
        ├── 2026-09-16-open-scene-vision-demo-design.md # 初始整体架构设计
        ├── 2026-09-26-event-storage-export-design.md  # 事件证据存储与导出设计
        └── 2026-09-27-portable-demo-api-settings-design.md # 可移植运行和 API 设置设计
```

## 本地运行目录（不提交 Git）

```text
.venv/                                 # 本机 Python 解释器与安装依赖
runs/
├── uploads/                            # 用户上传视频的本地副本
├── <分析运行目录>/                      # 每次运行的结果视频、截图、事件和规则过程
├── qwen/                               # Qwen 采样帧、检验报告、证据及缓存
├── settings/                           # Qwen 非密钥设置、调用额度数据库等
├── share/                              # 公网隧道工具、地址、进程编号和启动日志
├── ultralytics/                        # 项目内 YOLO 配置，避免用户目录权限问题
├── multi_person_source/                # 多人视频原始素材、预览和处理验证记录
└── 其他验证目录/                        # 测试临时文件、实际模型验收和诊断输出
```

API Key 存在操作系统凭据管理器中，不在上述源码或 Git 仓库内。Git 中包含代码、配置、测试、文档、上述 9 份示例视频、素材来源和下载脚本；不包含模型权重、用户上传内容及本机运行状态。`runs/` 内的下载原片与诊断副本不重复提交。

## 建议阅读顺序

先看 `app.py` 理解网页流程，再看 `src/pipeline/analysis_pipeline.py` 理解分析链路。新增小模型主要涉及 `models/`、`config/models.yaml`；新增事件判断主要涉及 `rules/` 和 `config/scenes/`；调整网页看 `ui/`，调整 DeepSeek/Qwen 行为看 `llm/` 与 `pipeline/qwen_pipeline.py`。
