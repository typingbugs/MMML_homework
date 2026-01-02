# 本地 AI 智能文献与图像管理助手（Local Multimodal AI Agent）

## 作者信息

北京交通大学 计算机科学与技术学院  
研究生课程《多模态机器学习》大作业  
姓名：柯劲帆  
学号：25120323

**演示脚本见 `example.sh`**

---

## 1. 项目概览

本项目是一个基于 Python 的**本地多模态知识库管理助手**，面向“论文 PDF + 图片素材”的两类数据，提供：

- 论文：添加/批量添加、基于向量库的语义检索、按主题自动归档
- 图片：添加/批量添加、以文搜图（也支持“以图搜图”）

项目采用模块化设计：**Embedding 模型（文本/图像）**与**向量数据库**可替换；默认使用 ChromaDB 作为本地持久化向量库。

CLI 统一入口：根目录 `main.py`。

### 1.1 功能概览

#### 论文（PDF）

- **语义搜索**：对查询文本生成向量，在 ChromaDB 中检索最相近的论文片段，并输出对应论文路径与主题。
- **单文件添加与自动归档**：`add_paper <pdf_path> --topics "A,B"`
	- 传入 `--topics`：论文会复制到 `data/papers/<topic>/` 下（可多主题）。
	- 不传 `--topics`：系统会基于向量库近邻结果做“主题投票”推断主题；若库为空则归为 `unknown`。
- **批量整理**：`add_paper <dir_path>` 对目录下（非递归）所有 `.pdf` 批量索引与归档。

说明：当前检索已返回 **论文内容片段**（chunk），但未提供页码级定位（如需页码，需要在切块时记录页号与偏移映射）。

#### 图片（Image）

- **以文搜图**：对自然语言查询生成向量，在 ChromaDB 中检索最匹配的图片条目，并输出图片路径与主题。
- **以图搜图（额外能力）**：当 `search_image <query>` 的 `query` 是本地图片路径时，会提取该图片向量作为查询向量进行相似检索。
- **单文件添加与自动归档**：`add_image <img_path> --topics "cat"`，将图片复制到 `data/images/<topic>/`，并写入向量库。
- **批量整理**：`add_image <dir_path>` 对目录下（非递归）所有图片（`.jpg/.jpeg/.png/.webp`）批量索引与归档。

---

## 2. 架构

### 2.1 模块化组件

- **Embeddings**：
	- 文本：OpenAI 兼容 embeddings / Ark
	- 图像：Ark 多模态 embeddings
- **VectorDB**：ChromaDB（本地持久化，无需额外服务）
- **Service Layer**：`PaperService` / `ImageService` 负责“加载 → 嵌入 → 入库 → 检索 → 归档”

### 2.2 文件结构

```
main.py                      # CLI 统一入口
config/config.yaml           # 模型与路径配置

app/
	paper_service.py           # 论文：切块、嵌入、入库、检索、归档
	image_service.py           # 图片：嵌入、入库、检索、归档

infrastructure/
	embeddings/
		text/                    # 文本嵌入：OpenAI兼容 / Ark
		image/                   # 图像嵌入：Ark 多模态 embeddings
	vector_db/
		chroma.py                # ChromaDB 持久化向量库封装

utils/
	pdf_loader.py              # PDF 读取（PyPDF2）
	image_loader.py            # 图片目录扫描
	config_loader.py           # YAML 配置加载

data/
	papers/                    # 归档后的论文
	images/                    # 归档后的图片
	vector_db/chroma/          # ChromaDB 持久化目录
	test_input/                # 示例输入数据（便于演示）
```

---

## 3. 安装与配置

### 3.1 Python 版本

建议 Python 3.8+。

### 3.2 安装依赖

```bash
pip install -r requirements.txt
```

依赖说明（与代码一致）：

- `chromadb`：本地持久化向量数据库
- `openai`：OpenAI 兼容 embeddings 客户端（也可对接兼容 API 的第三方服务）
- `volcengine-python-sdk[ark]`：火山 Ark 多模态 embeddings
- `PyPDF2`：PDF 文本提取
- `pyyaml`：读取 `config/config.yaml`
- `tqdm`：批量向量化进度条
- `python-dotenv`：从 `.env` 读取密钥（`main.py` 会调用 `load_dotenv()`）

### 3.3 环境变量（API Key）

在项目根目录创建 `.env`（或直接在系统环境变量中设置）：

```bash
# OpenAI 兼容 embeddings（论文文本向量化时使用）
OPENAI_API_KEY=your_key

# Ark 多模态 embeddings（图片向量化、以及可选的文本向量化时使用）
ARK_API_KEY=your_key
```

### 3.4 配置文件（`config/config.yaml`）

关键字段：

- `model.paper.text`：论文文本 embedding 的服务与模型（OpenAI 兼容接口）
- `model.image.text`：以文搜图的文本 embedding（可选 Ark / OpenAI 兼容）
- `model.image.image`：图片 embedding（Ark 多模态）
- `vector_db.persist_dir`：ChromaDB 持久化目录
- `file_dir.paper` / `file_dir.image`：归档后的论文/图片目录

---

## 4. 使用说明（CLI）

统一入口：`python main.py <command> ...`。

### 4.1 添加/分类论文

```bash
# 单个 PDF：显式指定主题（可逗号分隔多个主题）
python main.py add_paper "/path/to/paper.pdf" --topics "CV,NLP"

# 批量整理：传入目录（非递归，仅扫描该目录下的 .pdf）
python main.py add_paper "/path/to/pdf_dir" --topics "RL"

# 不提供 topics：自动主题推断（近邻投票），并归档到 data/papers/<topic>/
python main.py add_paper "/path/to/paper.pdf"
```

### 4.2 语义搜索论文

```bash
python main.py search_paper "Transformer 的核心架构是什么？" --top_k 5
```

输出包含：论文归档路径、主题、以及匹配到的内容片段（chunk）。

### 4.3 添加/分类图片

```bash
# 单张图片
python main.py add_image "/path/to/image.png" --topics "cat"

# 批量图片目录（非递归）
python main.py add_image "/path/to/image_dir" --topics "dog"
```

### 4.4 以文搜图（也支持以图搜图）

```bash
# 以文搜图
python main.py search_image "a photo of a cute dog" --top_k 5

# 以图搜图：query 是图片文件路径
python main.py search_image "/path/to/query.png" --top_k 5
```

---

## 5. Quick Demo

仓库提供了一个快速演示脚本：

```bash
bash example.sh
```

它会：清空 `data/` 下的归档与向量库目录，然后对 `data/test_input/` 的论文与图片做添加与检索演示。

---

## 6. 核心实现逻辑（对应代码）

### 6.1 论文管线（`PaperService`）

1. **读取 PDF**：使用 `PyPDF2` 提取全文文本（`utils/pdf_loader.py`）。
2. **切块（chunking）**：将全文按滑窗切分为多个片段（窗口 1024 字符，步长 500 字符）。
3. **向量化（embedding）**：对每个 chunk 调用文本 embedding 生成向量；如传入 `--topics`/标题，会在输入文本前添加 `Topics:`、`Title:` 前缀。
4. **主题推断（未指定 topics 时）**：对 chunk 的 embedding 做近邻检索并对主题投票；若无结果返回 `unknown`。
5. **持久化入库**：将 `documents + embeddings + metadatas` 写入 ChromaDB；metadata 记录归档后 `path`（可多主题）与 `topics`。
6. **归档保存**：论文以 `md5(原路径)+.pdf` 命名，复制到 `data/papers/<topic>/`。

### 6.2 图像管线（`ImageService`）

1. **图像向量化**：读取图片并 base64 编码，通过 Ark `multimodal_embeddings` 获取图像向量；如指定 `--topics`，会将 `Topics:` 文本与图片一起作为多模态输入。
2. **文本向量化（以文搜图）**：对文本 query 调用文本 embedding（OpenAI 兼容或 Ark）。
3. **检索**：在 ChromaDB 中按向量相似度检索，返回图片路径与主题。
4. **主题推断（未指定 topics 时）**：对图像向量做近邻检索并对主题投票；若无结果返回 `unknown`。
5. **归档保存**：图片以 `md5(原路径)+后缀` 命名，复制到 `data/images/<topic>/`。