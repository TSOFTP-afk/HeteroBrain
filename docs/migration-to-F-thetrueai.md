# 项目迁移文档：f:\项目\THE TRUE AI → F:\thetrueai

**迁移日期**：2026-07-26
**迁移执行者**：TRAE AI 助手
**用户授权**：用户在会话中明确选择"项目整体搬到 F:\thetrueai"

## 迁移原因

### 1. PowerShell 5.1 中文路径编码问题
- Windows PowerShell 5.1 在读取 `.ps1` 文件时，若文件无 BOM，则按系统 ANSI 代码页（中文 Windows 为 GBK）解析
- Write 工具写入的 `.ps1` 是 UTF-8 无 BOM，导致 `.ps1` 内的中文字符串字面量被当 GBK 误解析
- 表现：`Get-ChildItem` 返回空、`Test-Path` 报 "Illegal characters in path"、`Get-Content` 输出乱码
- 规避方法：.ps1 内容必须纯 ASCII，中文用 `([char]0xXXXX).ToString()` 拼接 —— 极其繁琐且易错

### 2. 编译/运行链路全程需 ASCII 路径
- llama.cpp 的 CMake、ninja、nvcc 对中文路径支持不稳定
- 已建立的 junction 链路（F:\hb_llama, F:\hb_build, F:\hb_models）虽绕开了源码/编译/模型路径，但工作区自身的中文路径仍影响脚本执行

### 3. 沙箱权限与路径无关，但操作一致性要求
- TRAE IDE 沙箱按工作区路径白名单允许 Write/Edit
- 编译产物若在工作区外（F:\hb_build），删除/修改操作被沙箱拦截
- 迁移到 F:\thetrueai 后，工作区全英文，且可考虑将 build 目录也纳入工作区内

## 迁移前布局

```
f:\项目\THE TRUE AI\          (中文路径，4116.8 MB, 9993 文件)
├── .git\                      (27.57 MB, 166 tracked files)
├── .trae\                     (0.26 MB, TRAE 会话数据)
├── configs\                   (配置文件)
├── data\                      (空)
├── docs\
│   ├── archive\               (3395.28 MB! 历史 SNN checkpoint/日志，已 git-ignore)
│   ├── migration_from_legacy.md
│   └── roadmap.md
├── legacy\                    (1.19 MB, 旧 SNN 代码)
├── models\
│   └── MiniCPM5-1B-Q4_K_M.gguf (656.19 MB, 已 git-ignore)
├── scripts\                   (0.02 MB, hb_*.ps1 系列)
├── src\                       (0.01 MB)
├── tests\                     (空)
├── third_party\               (空)
├── .gitignore
├── CMakeLists.txt
├── LICENSE
├── llama.cpp-master.zip       (36.23 MB)
├── PROJECT_MEMORY.md          (本次迁移前新建)
└── README.md

F:\hb_llama\                   (junction → llama.cpp 源码, 英文)
F:\hb_build\                   (编译产物, 英文, 在工作区外)
F:\hb_models\                  (模型 GGUF + Jinja 模板, 英文, 在工作区外)
```

## 迁移后布局

```
F:\thetrueai\                  (全英文, IDE 工作区)
├── .git\                      (从原路径迁来)
├── .trae\                     (从原路径迁来, TRAE 会话数据)
├── configs\
├── data\
├── docs\
│   ├── archive\               (3.4GB, 历史, 仍 git-ignore)
│   ├── migration_from_legacy.md
│   ├── migration-to-F-thetrueai.md  ← 本文件
│   └── roadmap.md
├── legacy\
├── models\
│   └── MiniCPM5-1B-Q4_K_M.gguf
├── scripts\
│   └── hb_*.ps1
├── src\
├── tests\
├── third_party\
├── .gitignore
├── CMakeLists.txt
├── LICENSE
├── llama.cpp-master.zip
├── PROJECT_MEMORY.md          ← 项目自带记忆文档
└── README.md

F:\hb_llama\                   (junction, 保持不变)
F:\hb_build\                   (编译产物, 保持不变, 仍英文)
F:\hb_models\                  (模型, 保持不变, 仍英文)
```

## 迁移方法

使用 `robocopy` 同盘复制：
- `/E` 包含空子目录
- `/COPY:DAT` 复制数据、属性、时间戳（不复制 ACL，避免权限问题）
- `/R:1 /W:1` 失败重试 1 次，等待 1 秒（避免卡死）
- `/MT:8` 8 线程并行
- `/NFL /NDL` 不打印每个文件/目录名（减少日志噪音）
- `/NP` 不显示进度百分比

## 验证标准

1. 文件总数一致（9993）
2. 总大小一致（4116.8 MB ± 0.01MB 容差）
3. `.git` 完整（166 tracked files）
4. `git status` 在新路径下干净（除新增 PROJECT_MEMORY.md 和本迁移文档）
5. `MiniCPM5-1B-Q4_K_M.gguf` 大小 = 688065920 字节
6. 关键脚本（hb_*.ps1）可执行

## 迁移后操作

1. 用户重启 TRAE IDE
2. 在新路径 `F:\thetrueai\` 打开工作区
3. TRAE 自动在新路径下创建 memory 文件夹 `c:\Users\26455\.trae-cn\memory\projects\-F----thetrueai\`
4. 助手在新会话中优先读取 `PROJECT_MEMORY.md` 恢复上下文
5. 确认 `F:\hb_llama`, `F:\hb_build`, `F:\hb_models` 三个英文路径仍可用
6. 继续未完成任务：编译 llama-cli + 用 Jinja 模板测试中文推理

## 旧路径处理

迁移验证通过后，旧路径 `f:\项目\THE TRUE AI\` 暂时保留不删除，作为回滚保险。
用户在新路径稳定运行 1-2 周后可手动删除旧路径。

## 不迁移的内容

- `F:\hb_llama` (junction，指向 third_party\llama.cpp，已在英文路径)
- `F:\hb_build` (编译产物，已在英文路径)
- `F:\hb_models` (模型文件，已在英文路径)
- `c:\Users\26455\.trae-cn\memory\` (TRAE memory 系统，按工作区路径独立)

这些路径迁移后保持不变，脚本中的 `F:\hb_*` 引用无需修改。

## 已知遗留问题

1. `.gitignore` 中的中文注释在 PS 5.1 下显示乱码（不影响 git 功能，git 用 UTF-8 解析）
2. `docs/archive/` 占 3.4GB，若未来不需要历史 SNN checkpoint，可单独清理
3. `llama.cpp-master.zip` (36MB) 已解压到 `third_party\llama.cpp`，zip 本身可删除但暂保留
