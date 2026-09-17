# 模型与样例选择记录

本文记录答辩 Demo 实际使用的模型、素材、许可证边界、文件校验和与本机验证结果。性能数据只代表当前笔记本、当前软件版本和固定样例，不作为正式精度指标。

## 1 边防通用检测基线

### 1.1 模型信息

- 模型：Ultralytics YOLO11n Detect，COCO 预训练权重。
- 权重来源：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>
- Python 包版本：`ultralytics 8.4.154`。
- 推理框架：`torch 2.14.0+cpu`，运行检查结果为 `cuda_available=False`。
- 权重大小：5,613,764 字节。
- 权重 SHA-256：`0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`。
- 本期使用类别：`person`、`car`、`motorcycle`、`bus`、`truck`。
- 许可证：Ultralytics 软件与模型采用 AGPL-3.0 或商业许可证双许可。本 Demo 按教学与研究验证使用；若转为闭源产品或商业部署，必须重新进行许可证评估并取得合适授权。

### 1.2 固定样例视频

- 文件：OpenCV 官方样例 `samples/data/vtest.avi`。
- 来源：<https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi>
- 仓库许可证：Apache-2.0。
- 文件大小：8,131,690 字节。
- 视频 SHA-256：`45CDDC9490BE69345CBDAB64CA583BE65987E864CA408038E648DB99E10516CF`。
- 元数据：768×576、10 FPS、795 帧；固定监控视角，画面包含多名行人与车辆。

### 1.3 CPU 实测结果

测试范围为同一视频的前 100 帧，置信度阈值为 0.25，逐帧处理且不抽帧。`effective_fps` 包含模型推理、画框和视频写出耗时。

| 输入尺寸 | 帧数 | 检测框总数 | 平均推理耗时 | 完整流程速度 | 输出文件 |
|---:|---:|---:|---:|---:|---|
| 416 | 100 | 739 | 84.650 ms/帧 | 10.463 FPS | `runs/validation/border_vtest_100f.mp4` |
| 640 | 100 | 706 | 99.185 ms/帧 | 9.265 FPS | `runs/validation/border_vtest_100f_640.mp4` |

当前 CPU 基线优先采用输入尺寸 416。它在本次固定测试中的平均推理耗时更低；检测框总数只能用于检查运行差异，不能替代准确率、召回率或人工标注评估。抽帧策略留到视频采样模块实现后单独验证。

### 1.4 已验证边界

- 已验证：CPU 加载模型、人员与车辆类别过滤、统一框转换、标注视频写出、结果视频重新读取。
- 尚未实现：目标跟踪、区域规则、跨线事件、告警去重以及 Streamlit 集成。

## 2 火焰烟雾模型

### 2.1 固定样例

- 正样本：Wikimedia Commons `Fire burning.ogv`，作者 Zouavman Le Zouave，可按 CC BY-SA 4.0 使用。
- 正样本来源：<https://commons.wikimedia.org/wiki/File:Fire_burning.ogv>
- 正样本元数据：640×480、25 FPS、279 帧，约 11 秒。
- 正样本 SHA-256：`B9C0944E78CC68300D11BC79D8ED5BD4BDEC5D1B3823FACF0BE3E0BDBC94F36A`。
- 负样本：复用 OpenCV `vtest.avi` 的前 100 帧，画面包含行人、车辆、草地和建筑，但没有火焰或烟雾。

### 2.2 候选模型

| 候选 | 架构与类别 | 权重大小 | 权重 SHA-256 | 许可证 | 兼容性 |
|---|---|---:|---|---|---|
| `rabahdev/fire-smoke-yolov8n` | YOLOv8n；`smoke`、`fire` | 6,229,802 字节 | `B91633799CEB052C814B4F8B77A37EFC9A40F002D528DF97D74463585FA4F28F` | AGPL-3.0 | 类别元数据完整，当前 Ultralytics 可直接加载 |
| `pyronear/yolo11s_sensitive-detector` | YOLO11s；模型卡说明为野火烟雾 | 19,225,626 字节 | `A9BFA11C559E4B221C43BB24A0EC0857BC75EC653283BA6146AEF44C088DA51A` | Apache-2.0；运行时仍使用 Ultralytics | 权重内类别名为通用 `item`，需要额外映射 |

主候选固定版本：Hugging Face 提交 `13017fe8af477c25f5298d168e2dfede4b000753`。备选固定版本：提交 `56d67d6f4d7eab7feed1bebf656f8f73de26ff03`。

### 2.3 同片段 CPU 对比

两个候选均使用输入尺寸 416、置信度 0.25，并处理相同的正样本前 100 帧和负样本前 100 帧。

| 候选 | 正样本有框帧 | 正样本框数 | 正样本速度 | 负样本有框帧 | 负样本框数 | 负样本速度 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n 双类别 | 100/100 | `fire` 107，`smoke` 0 | 15.071 FPS | 0/100 | 0 | 16.270 FPS |
| Pyronear YOLO11s 烟雾 | 0/100 | 0 | 9.388 FPS | 0/100 | 0 | 9.408 FPS |

主候选额外处理了完整 279 帧正样本：279 帧均出现检测，共产生 358 个 `fire` 框，完整流程速度为 16.606 FPS。抽查中间帧后确认检测框覆盖真实火焰区域。

### 2.4 选择结论

选择 `rabahdev/fire-smoke-yolov8n` 作为本 Demo 主模型，原因是权重更小、类别元数据完整、同时声明火焰与烟雾、当前火焰正样本响应稳定，并且本机 CPU 速度明显更高。

不将 Pyronear 模型作为当前备用模型：它面向远距离野火烟雾，权重内类别语义不完整，而且在本次近景火焰样本上没有响应。当前固定素材只验证了火焰能力，尚未证明主模型的烟雾检测效果；在取得合适烟雾正样本并完成验证前，答辩口径应收敛为“火焰检测已验证，烟雾检测为模型支持但未验收”。
