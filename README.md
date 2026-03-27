# 智能问答系统 项目

一个基于 Flask 和 LangChain 的轻量级 RAG（检索增强生成）应用，支持文档上传、知识库管理、智能问答等功能。

## 功能特性

- 📄 支持多种文档格式上传（PDF、DOCX、TXT、MD）
- 🗄️ 支持本地存储和 MinIO 对象存储
- 🔍 支持多种向量数据库（Chroma、Milvus）
- 🤖 支持多种 LLM 模型（DeepSeek、Gemini、Ollama）
- 💬 智能问答和对话管理
- 🔐 用户认证和权限管理
- 📊 知识库管理和文档检索

## 环境要求

- Python 3.13 或更高版本
- MySQL 5.7+ 或 8.0+
- uv 包管理器（推荐）
- Docker 和 Docker Compose（可选，用于 Milvus）

## 快速启动

### 1. 安装依赖

#### 使用 uv（推荐）

```bash
# 安装 uv（如果尚未安装）
pip install uv

# 使用 uv 安装依赖
uv sync

# 确保mysql数据库已运行且已创建rag数据库
# 使用 uv 启动项目
uv run main.py
```

### 2. 配置数据库

#### 方式一：使用本地 MySQL

```bash
# 登录 MySQL
mysql -u root -p

# 创建数据库
CREATE DATABASE rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# 创建用户（可选）
CREATE USER 'rag_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON rag.* TO 'rag_user'@'localhost';
FLUSH PRIVILEGES;
```

#### 方式二：使用 Docker 快速启动 MySQL

```bash
docker run --name rag-mysql \
  -e MYSQL_ROOT_PASSWORD=123456 \
  -e MYSQL_DATABASE=rag \
  -p 3306:3306 \
  -d mysql:8.0
```

### 3. 启动向量数据库

#### 使用 Chroma（默认，无需额外配置）

Chroma 会自动在本地创建，无需额外配置。

#### 使用 Milvus（可选）

```bash
# 启动 Milvus 及其依赖服务
docker-compose up -d

# 检查服务状态
docker-compose ps
```

### 4. 启动应用

```bash
# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 启动应用
uv run main.py
```

启动成功后，你会看到类似以下的输出：

```
INFO - 正在启动RAG服务器在0.0.0.0:5000
 * Running on http://0.0.0.0:5000
```

### 5. 访问应用

在浏览器中打开：`http://localhost:5000`

## 项目结构

```
├── app/                    # 应用主目录
│   ├── blueprints/        # Flask 蓝图
│   │   ├── auth.py       # 认证相关路由
│   │   ├── chat.py       # 聊天相关路由
│   │   ├── document.py   # 文档相关路由
│   │   ├── knowledgebase.py  # 知识库相关路由
│   │   ├── settings.py   # 设置相关路由
│   │   └── utils.py      # 工具路由
│   ├── models/           # 数据模型
│   ├── services/         # 业务逻辑层
│   ├── templates/        # HTML 模板
│   └── utils/           # 工具函数
├── chroma_db/           # Chroma 向量数据库存储
├── logs/               # 日志文件
├── storages/           # 文件存储目录
├── main.py            # 应用入口
├── pyproject.toml     # 项目配置
└── docker-compose.yml # Docker Compose 配置
```

## 开发说明

### 添加新的依赖

```bash
uv add package-name
```

## 许可证

MIT License
