# Demo 测试与验收记录

## 1. 验收范围

验收针对本项目已实现的 CPU 本地视频 Demo，覆盖：场景配置、模型选择、视频处理、跟踪、规则、事件、截图、结果视频、结构化导出、Web 页面以及无 DeepSeek Key 的离线路径。

公开视频上的结果只证明当前素材和当前环境下流程可运行，不能替代正式准确率、误报率、事件检出率或并发性能报告。

## 2. 自动测试

命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

2026-09-26 最终交付分支实测结果：

```text
97 passed in 3.97s
```

测试覆盖：

- SceneSpec 字段、坐标、模型、目标类别、规则和参数白名单。
- 模型注册表、YOLO/火灾适配器输出归一化。
- 视频元数据、损坏文件、抽帧、写出、释放和 H.264 转码。
- 跟踪 ID、禁区进入/离开、滞留、冷却和火灾连续帧状态。
- 事件去重、编号、截图、JSON/CSV/摘要导出和中文 Windows 路径。
- DeepSeek 超时/重试、无效输出、未知字段和离线回退。
- 主流水线、进度回调、正常停止和资源释放。
- Streamlit 控件、事件表、安全截图画廊和下载入口。
- 验收产物的跨文件一致性、截图可读性和 H.264 检查。

测试时 FFmpeg 可能对故意构造的损坏 MP4 输出 `moov atom not found`；该行是负向用例的预期解码日志，对应测试已通过。

## 3. 真实 CPU 重复验收

环境：Windows 10 `10.0.19045`，Python `3.11.9`，仅 CPU。验收前从当前 PowerShell 删除 `DEEPSEEK_API_KEY`，因此全部使用离线模板，DeepSeek 网络请求数为 `0`。

命令：

```powershell
Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\run_acceptance.py --repeat 3
```

报告位置：`runs/acceptance/20260926-184241/acceptance_report.json`。

| 场景 | 轮次 | 模型 | 帧数 | 事件 | 截图 | 流水线耗时 | 总耗时 | 结果 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 边防 | 1 | `yolo_general` | 120 | 3 | 3 | 11.403 s | 11.814 s | 通过 |
| 火灾 | 1 | `fire_smoke` | 275 | 2 | 1 | 13.961 s | 14.901 s | 通过 |
| 边防 | 2 | `yolo_general` | 120 | 3 | 3 | 7.682 s | 8.115 s | 通过 |
| 火灾 | 2 | `fire_smoke` | 275 | 2 | 1 | 14.761 s | 15.636 s | 通过 |
| 边防 | 3 | `yolo_general` | 120 | 3 | 3 | 7.237 s | 7.692 s | 通过 |
| 火灾 | 3 | `fire_smoke` | 275 | 2 | 1 | 14.357 s | 15.375 s | 通过 |

结论：6/6 次运行通过。边防每轮均生成 3 个事件及 3 张截图；火灾每轮均生成 1 个疑似事件和 1 个确认事件，确认事件生成 1 张截图。全部 `events.json`、`events.csv` 和 `summary.json` 事件数一致，结果视频可读且包含 `avc1` H.264 标识。

## 4. 备用结果

最后一轮已通过的两个场景被复制到：

```text
runs/backup/20260926-184241/
├── border/
│   ├── config.json
│   ├── events.json
│   ├── events.csv
│   ├── summary.json
│   ├── snapshots/
│   └── result.mp4
└── fire/
    ├── config.json
    ├── events.json
    ├── events.csv
    ├── summary.json
    ├── snapshots/
    └── result.mp4
```

该目录只存在本机且被 Git 忽略。重新运行验收会新建一个时间目录，不覆盖旧证据。

## 5. 自动验收器的通过条件

每次真实运行必须同时满足：

1. `config.json`、`events.json`、`events.csv`、`summary.json` 和 `result.mp4` 全部存在。
2. JSON 事件列表、CSV 数据行和摘要中的 `event_count` 相等。
3. 事件数不少于该场景规定的最小值。
4. 截图路径必须位于本次 `snapshots/` 内，且图像能成功解码。
5. `result.mp4` 能读取至少一帧，帧数大于 0，文件包含 H.264 `avc1` 标识。
6. 任一检查失败时 CLI 返回非 0 退出码，不生成“验收通过”结论。

## 6. 人工验收清单

- [ ] 按用户手册从新 PowerShell 启动 Streamlit。
- [ ] 上传 `border_demo.avi`，完成边防人员禁区流程。
- [ ] 确认结果视频中有框、类别和 ID，无置信度。
- [ ] 确认边防事件表、截图和 5 个下载入口。
- [ ] 上传 `fire_demo.avi`，完成火灾流程。
- [ ] 确认“疑似火灾”和“确认火灾”，且只有确认事件有截图。
- [ ] 下载一份 JSON 和 CSV，核对事件数与页面一致。
- [ ] 在无 Key 状态下重新生成离线场景配置。

这一节保留给用户最终人工验收；自动测试和 6 次真实后端运行已通过，但不替代用户对页面视觉效果和操作感受的确认。
