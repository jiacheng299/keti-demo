# 可移植 Demo 与 DeepSeek 凭据设置设计

## 1. 目标

将当前 Demo 调整为一个可从 GitHub 直接获取的完整项目。新用户克隆仓库后，只需安装 `requirements.txt` 中的依赖并启动 Streamlit，即可使用内置的两个模型权重和两段示例视频完成边防、火灾分析。

同时在 Web 页面中增加 DeepSeek API 设置。API Key 使用操作系统凭据管理器持久保存，不保存到项目文件、日志、Git 或运行结果中。

## 2. 用户成功标准

新用户执行：

```powershell
git clone https://github.com/jiacheng299/keti-demo.git
cd keti-demo
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

之后必须能够：

1. 在页面中看到两段内置示例视频及其适用场景说明。
2. 上传示例视频，不再从外部下载权重或素材，完成边防或火灾流程。
3. 在页面中保存 DeepSeek API Key，重启页面和电脑后仍可使用。
4. 在页面中删除已保存的 Key。
5. 不配置 Key 时继续通过离线模板完成分析。

## 3. 交付内容

### 3.1 仓库内置资源

从 `.gitignore` 中仅放行四个已审核文件：

- `models/border/yolo11n.pt`，5,613,764 字节。
- `models/fire/fire_smoke_yolov8n.pt`，6,229,802 字节。
- `assets/demo_videos/border_demo.avi`，约 12 秒。
- `assets/demo_videos/fire_demo.avi`，约 11 秒。

原始长视频、其他权重、运行结果和用户上传文件仍保持忽略。不引入 Git LFS，因为每个文件都远小于 GitHub 的单文件上限，且希望新用户不需要额外客户端。

仓库增加第三方资源告知，清楚记录：

- Ultralytics YOLO11n 和火焰/烟雾权重的 AGPL-3.0 边界。
- OpenCV 示例视频的 Apache-2.0 来源。
- Wikimedia Commons 火焰视频的 CC BY-SA 4.0 来源与归属。
- 短片来自已记录源视频，未改变原许可边界。

### 3.2 安装和启动契约

`requirements.txt` 保持为唯一必需的 pip 依赖清单，新增 `keyring` 用于操作系统凭据保存。不发布 PyPI 包，不新增自定义命令行入口。

标准启动命令保持为：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

页面展示示例视频的本地路径和场景建议。用户仍通过原有上传控件选择视频，不为页面增加第二条分析流水线。

## 4. DeepSeek 凭据设计

### 4.1 存储边界

新增独立凭据服务，对外提供小而稳定的接口：

```python
load_api_key() -> str | None
save_api_key(api_key: str) -> None
delete_api_key() -> None
```

生产实现使用 `keyring`，固定服务名为 `keti-demo-deepseek`，账户名为 `api-key`。Windows 上密钥保存在系统凭据管理器；其他系统由 `keyring` 选择可用的本机后端。

密钥持续保留，直到用户主动清除，因此满足“至少一周”的需求。系统不设自动过期，避免在用户不知情时中断。

### 4.2 密钥优先级

解析 DeepSeek 场景时使用以下优先级：

1. 当前页面会话中刚刚保存或已加载的操作系统凭据。
2. `DEEPSEEK_API_KEY` 环境变量，作为服务器和自动化运行的备用入口。
3. 无密钥时使用离线模板。

页面不修改 `os.environ`。凭据服务返回的 Key 通过显式参数传给 `DeepSeekClient(api_key=...)`，防止全局进程状态污染。

### 4.3 页面交互

在“运行状态”区域下方增加“DeepSeek API 设置”展开区：

- 密码型输入框，不回显已保存 Key。
- `保存 API Key` 按钮。
- `清除已保存 Key` 按钮。
- `已安全保存` / `未保存` / `凭据后端不可用` 状态。

保存前对输入执行最小校验：去除首尾空白后不得为空。页面不打印 Key，不显示 Key 长度或前后缀。保存成功后清空输入控件，并更新状态。

不在保存时自动发起计费的 DeepSeek 请求。真实 API 请求仍只在用户选择“DeepSeek 解析”并点击“生成场景配置”时发生。

### 4.4 异常处理

- 操作系统凭据后端不可用：页面显示错误，不降级为明文文件保存；环境变量和离线模板仍可用。
- 保存失败：输入框保留，方便用户重试，但错误信息不包含 Key。
- 删除失败：不宣称已清除，页面显示可重试的简短错误。
- DeepSeek 鉴权、超时或输出错误：保持已有模板回退行为，不自动删除凭据。

## 5. 组件与数据流

```text
Streamlit API 设置
        |
        v
CredentialService -----> 操作系统凭据管理器
        |
        v
resolve_scene_spec(api_key=...)
        |
        v
DeepSeekClient(api_key=...)
        |
        v
SceneParser -> 本地白名单校验 -> SceneSpec
        |
        v
现有 AnalysisPipeline
```

凭据服务只管理 Key，不负责 API 请求。`DeepSeekClient` 只使用显式传入的 Key 或环境变量，不直接依赖 `keyring`。Streamlit 服务层负责组合两者。

## 6. 文件变更边界

预计新增或修改：

- `.gitignore`：仅放行四个可分发资源。
- `requirements.txt`：新增固定范围的 `keyring` 依赖。
- `src/security/credential_store.py`：凭据存储抽象和 `keyring` 实现。
- `src/ui/app_service.py`：接受显式 API Key 并提供示例资源元数据。
- `app.py`：DeepSeek 设置区和示例视频提示。
- `tests/`：凭据服务、API Key 传递、页面控件、内置资源完整性测试。
- `THIRD_PARTY_NOTICES.md`、`README.md`、`docs/04_user_guide.md`、`docs/05_test_and_acceptance.md`：安装、资源、许可、凭据与验收说明。

不修改模型、规则、跟踪、事件或主流水线契约。

## 7. 测试设计

使用测试替身凭据后端，绝不在自动测试中读写用户真实凭据管理器。依次验证：

1. 保存时去除首尾空白，空 Key 被拒绝。
2. 保存后可读取，删除后返回未配置。
3. 后端异常被转换为不包含 Key 的领域错误。
4. 已保存 Key 被显式传入 `DeepSeekClient`，环境变量仍可作为备用。
5. Streamlit 页面包含密码输入、保存、清除和状态控件，不显示传入的 Key。
6. 四个分发资源存在，大小和 SHA-256 与文档记录一致。
7. 无 Key 的离线模板路径仍不会发起网络请求。

最终验收包含：

- 全量 pytest、`pip check`、`compileall` 和 Git 密钥扫描。
- 在新建虚拟环境中执行 `pip install -r requirements.txt`。
- 使用仓库内置资源完成边防和火灾真实 CPU 验收。
- 人工验证页面保存/清除凭据状态，测试后删除测试 Key。
- 检查 Git 暂存区仅包含指定权重和示例视频，不包含其他二进制文件或密钥。

## 8. GitHub 集成

开发在 `recall/portable-demo-api-settings` 分支完成。通过最终验收后：

1. 将本分支提交为一个语义完整的功能提交。
2. 将 `recall/portable-demo-api-settings` 快进合并到本地 `main`。
3. 将 `main` 推送到 `origin/main`。
4. 从 GitHub 远程信息确认新提交与四个分发资源已上传。

推送前不重写远程历史，不强制推送。

## 9. 明确非目标

- 不将 Demo 发布到 PyPI。
- 不使用 Git LFS。
- 不保存 API endpoint、模型名或其他供应商凭据。
- 不自动调用 DeepSeek 测试 Key 有效性，避免未预期计费。
- 不将 Key 写入 `.env`、Streamlit secrets、YAML、JSON 或数据库。
- 不改变已冻结的边防、火灾算法和事件行为。
