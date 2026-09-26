# 事件存储与导出设计

## 1. 目标

本阶段将规则层产生的 `EventCandidate` 转换为可展示、可追溯、可导出的正式 `Event`，并生成事件截图、JSON、CSV、场景配置和运行摘要。

本阶段同时服务边防与火灾场景，不包含完整视频分析流水线和 Web 页面接入。

## 2. 非目标

- 不生成结果视频。
- 不实现 Streamlit 页面。
- 不使用数据库。
- 不实现历史运行目录清理。
- 不打包 ZIP 下载文件。
- 不在事件管理层重复实现规则引擎的时间冷却。

## 3. 模块划分

采用两个独立模块：

- `EventManager`：正式事件生成、精确去重、事件编号和确认事件截图。
- `EventExporter`：配置、事件、CSV 和摘要文件写出。

数据流：

```text
EventCandidate + 当前视频帧
              |
              v
        EventManager
              |
              v
            Event
              |
              v
        EventExporter
              |
              v
config.json / events.json / events.csv / summary.json / snapshots/
```

## 4. EventManager

### 4.1 接口

```python
manager = EventManager(
    run_dir=run_dir,
    scene_type=scene_type,
    model_id=model_id,
)

event = manager.add(candidate, frame)
```

`add()` 返回：

- 新事件：返回 `Event`。
- 精确重复候选：返回 `None`。

### 4.2 正式事件字段

`EventManager` 从 `EventCandidate` 复制：

- `event_type`
- `target_class`
- `track_id`
- `timestamp_seconds`
- `confidence`
- `trigger_rule`
- `alert_status`

由管理器补充：

- `event_id`
- `scene_type`
- `model_id`
- `snapshot_path`

### 4.3 事件编号

每个 `EventManager` 实例从 1 开始递增：

```text
evt-000001
evt-000002
evt-000003
```

每次运行使用独立目录，因此编号只要求在当前运行内唯一。

编号只在事件创建成功后递增。确认事件截图失败时，不登记去重键，也不消耗编号，允许调用方修复问题后重试。

### 4.4 精确去重

去重键：

```python
(
    candidate.frame_id,
    candidate.event_type,
    candidate.track_id,
    candidate.trigger_rule,
)
```

同一个候选对象被重复提交时，只生成一条正式事件。

规则时间冷却继续由 `RuleEngine` 负责。`EventManager` 不根据时间窗口再次过滤，避免两层冷却产生不同结果。

### 4.5 截图规则

- `alert_status == "confirmed"`：保存 JPEG 截图。
- `alert_status == "suspected"`：不保存截图，`snapshot_path=None`。

截图目录：

```text
runs/<run_id>/snapshots/
```

确认事件截图名称：

```text
snapshots/<event_id>.jpg
```

`snapshot_path` 保存相对运行目录的路径，例如：

```text
snapshots/evt-000002.jpg
```

路径统一使用 `/` 分隔，保证 JSON、CSV 和不同操作系统上的展示一致。

截图写入失败时抛出明确异常，不生成带有无效截图路径的事件。

## 5. EventExporter

### 5.1 接口

```python
exporter.export(
    run_dir=run_dir,
    config=scene_spec,
    events=events,
    metrics=metrics,
)
```

导出器创建运行目录，但不删除或覆盖其他运行目录。

导出前先在内存中完成全部 JSON 和 CSV 序列化。每个文件先写入同目录临时文件，全部临时文件写入成功后，再用 `os.replace` 逐个原子替换正式文件。序列化或临时写入失败时，清理临时文件并保留上一次导出结果；正式替换阶段失败时报错，调用方可重试。四个独立文件不声称具有整组事务原子性。

### 5.2 config.json

保存本次实际使用的 `SceneSpec`。格式要求：

- UTF-8。
- 中文不转义。
- 两空格缩进。

### 5.3 events.json

保存完整事件数组。字段来源于 `Event.model_dump(mode="json")`。

格式要求：

- UTF-8。
- 中文不转义。
- 两空格缩进。
- 事件顺序与传入列表一致。

### 5.4 events.csv

字段顺序固定为：

```text
event_id
scene_type
event_type
target_class
track_id
timestamp_seconds
confidence
trigger_rule
snapshot_path
alert_status
model_id
```

格式规则：

- UTF-8 with BOM，保证 Windows Excel 正确显示中文。
- `None` 写为空字符串。
- `timestamp_seconds` 固定保留三位小数。
- `confidence` 保留原始数值的稳定字符串表示。
- 行顺序与 `events.json` 一致。

JSON 与 CSV 必须具有相同事件数量和相同核心字段值。

### 5.5 summary.json

保存：

```json
{
  "event_count": 2,
  "confirmed_count": 1,
  "suspected_count": 1,
  "event_type_counts": {
    "fire_suspected": 1,
    "fire_confirmed": 1
  },
  "metrics": {
    "processed_frames": 100,
    "elapsed_seconds": 8.42
  }
}
```

事件类型统计按事件首次出现顺序写入。`metrics` 原样保存调用方提供的可序列化指标。

## 6. 运行目录

本阶段形成：

```text
runs/<run_id>/
├── config.json
├── events.json
├── events.csv
├── summary.json
└── snapshots/
    └── evt-000002.jpg
```

`result.mp4` 由后续完整分析流水线生成，不属于本阶段。

## 7. 错误处理

- 输入帧不是三通道 BGR 图像：拒绝确认事件截图。
- 截图编码或写入失败：抛出明确异常。
- 配置或指标不能 JSON 序列化：写文件前失败并保留已存在文件。
- 空事件列表：仍生成带表头的 CSV、空数组 JSON 和零计数摘要。
- 输出目录已存在：允许更新本次运行的导出文件，不影响其他运行目录。

## 8. 测试设计

### 8.1 EventManager

- 候选事件转换为字段完整的正式事件。
- 事件编号按当前运行递增。
- 相同候选事件重复添加时返回 `None`。
- 疑似事件不创建截图。
- 确认事件创建可重新读取的 JPEG。
- 边防事件保留 Track ID。
- 火灾事件允许 Track ID 为空。
- 无效帧或截图失败返回明确错误。

### 8.2 EventExporter

- 生成 `config.json`、`events.json`、`events.csv` 和 `summary.json`。
- JSON 与 CSV 事件数量一致。
- JSON 与 CSV 核心字段一致。
- 时间固定保留三位小数。
- CSV 可用 `utf-8-sig` 读取。
- 摘要计数与事件列表一致。
- 空事件列表仍生成合法文件。
- 两个运行目录互不覆盖。

## 9. 完成标准

- 所有新增测试通过。
- 完整测试套件无回归。
- 临时运行目录中能生成全部规定文件。
- 确认事件截图可由 OpenCV 重新读取。
- 疑似事件不生成截图。
- JSON、CSV 和摘要的事件数量一致。
- 文档明确记录实际验证结果与未实现边界。
