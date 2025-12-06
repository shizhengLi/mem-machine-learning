# MemMachine架构设计与核心概念

## 项目概述

MemMachine是一个开源的AI智能体记忆层，为AI应用提供短期记忆、长期记忆和个性化记忆能力，使AI智能体能够学习、适应和记住。基于对源代码的深入分析，本文将详细介绍MemMachine的架构设计和核心概念。

## 技术栈分析

### 核心依赖（基于pyproject.toml）

```python
dependencies = [
    "alembic>=1.17.1",          # 数据库迁移工具
    "asyncpg>=0.31.0",           # PostgreSQL异步驱动
    "aiosqlite>=0.20.0",         # SQLite异步驱动
    "boto3>=1.40.40",            # AWS SDK
    "fastapi>=0.116.1",          # Web框架
    "instructor[bedrock]>=1.12.0", # LLM交互框架
    "langchain-aws>=0.2.33",     # AWS LangChain集成
    "neo4j>=5.28.2",             # Neo4j图数据库驱动
    "nltk>=3.9.1",               # 自然语言处理库
    "openai>=1.104.2",           # OpenAI API
    "pgvector>=0.4.1",           # PostgreSQL向量扩展
    "prometheus-client>=0.22.1", # 监控指标
    "pydantic>=2.11.7",          # 数据验证
    "sqlalchemy>=2.0.43",        # ORM框架
    "uvicorn>=0.35.0",           # ASGI服务器
    "greenlet>=3.2.4",           # 协程支持
]
```

### 项目结构

```
src/memmachine/
├── main/                    # 主协调器
├── common/                  # 通用组件
│   ├── configuration/       # 配置管理
│   ├── episode_store/       # 对话存储
│   ├── embedder/           # 嵌入器
│   ├── filter/             # 过滤器
│   ├── language_model/     # 语言模型
│   ├── metrics_factory/    # 指标工厂
│   ├── resource_manager/   # 资源管理
│   └── session_manager/    # 会话管理
├── episodic_memory/         # 情景记忆系统
├── semantic_memory/         # 语义记忆系统
├── server/                  # REST API服务器
└── rest_client/            # 客户端SDK
```

## 核心数据模型

### Episode（对话条目）模型

基于`src/memmachine/common/episode_store/episode_model.py`的实际实现：

```python
from enum import Enum
from pydantic import AwareDatetime, BaseModel, JsonValue

class ContentType(Enum):
    """对话内容类型枚举"""
    STRING = "string"   # 字符串内容
    # 未来可扩展：'vector', 'image'等

class EpisodeType(Enum):
    """对话条目类型枚举"""
    MESSAGE = "message"  # 消息类型
    # 未来可扩展：'thought', 'action'等

class EpisodeEntry(BaseModel):
    """创建新对话条目的数据结构"""
    content: str                    # 对话内容
    producer_id: str               # 生产者ID
    producer_role: str             # 生产者角色
    produced_for_id: str | None    # 目标用户ID
    episode_type: EpisodeType | None = None
    metadata: dict[str, JsonValue] | None = None
    created_at: AwareDatetime | None = None

class Episode(BaseModel):
    """存储在历史记录中的完整对话消息模型"""
    uid: EpisodeIdT                 # 唯一标识符
    content: str                    # 消息内容
    session_key: str                # 会话键
    created_at: AwareDatetime       # 创建时间
    producer_id: str                # 生产者ID
    producer_role: str              # 生产者角色
    produced_for_id: str | None     # 目标用户ID
    sequence_num: int = 0           # 序列号
    episode_type: EpisodeType = EpisodeType.MESSAGE
    content_type: ContentType = ContentType.STRING
    filterable_metadata: dict[str, FilterablePropertyValue] | None = None
    metadata: dict[str, JsonValue] | None = None

    def __hash__(self) -> int:
        """基于UID的哈希函数，用于去重操作"""
        return hash(self.uid)
```

### SemanticFeature（语义特征）模型

基于`src/memmachine/semantic_memory/semantic_model.py`的实际实现：

```python
from enum import Enum
from pydantic import BaseModel, InstanceOf

class SemanticCommandType(Enum):
    """语义内存操作类型"""
    ADD = "add"
    DELETE = "delete"

class SemanticCommand(BaseModel):
    """LLM发出的语义特征变更指令"""
    command: SemanticCommandType
    feature: str
    tag: str
    value: str

class SemanticFeature(BaseModel):
    """语义记忆条目模型"""

    class Metadata(BaseModel):
        """语义特征的存储元数据"""
        citations: list[EpisodeIdT] | None = None  # 引用的对话条目
        id: FeatureIdT | None = None               # 特征ID
        other: dict[str, Any] | None = None        # 其他元数据

    set_id: SetIdT | None = None    # 特征集ID
    category: str                  # 特征类别
    tag: str                      # 特征标签
    feature_name: str             # 特征名称
    value: str                    # 特征值
    metadata: Metadata = Metadata()

    @staticmethod
    def group_features(
        features: list["SemanticFeature"],
    ) -> dict[tuple[str, str, str], list["SemanticFeature"]]:
        """按类别、标签、特征名称分组特征"""
        grouped_features: dict[tuple[str, str, str], list[SemanticFeature]] = {}

        for f in features:
            key = (f.category, f.tag, f.feature_name)
            if key not in grouped_features:
                grouped_features[key] = []
            grouped_features[key].append(f)

        return grouped_features
```

## MemMachine核心类

### 主协调器架构

基于`src/memmachine/main/memmachine.py`的实际实现：

```python
import asyncio
from enum import Enum
from typing import Protocol
from pydantic import BaseModel, InstanceOf

class MemoryType(Enum):
    """记忆类型枚举"""
    Semantic = "semantic"
    Episodic = "episodic"

class MemMachine:
    """MemMachine主类 - 记忆系统协调器"""

    class SessionData(Protocol):
        """会话数据协议"""
        @property
        def session_key(self) -> str:
            """唯一会话标识符"""
            raise NotImplementedError

        @property
        def user_profile_id(self) -> str | None:
            raise NotImplementedError

        @property
        def role_profile_id(self) -> str | None:
            raise NotImplementedError

        @property
        def session_id(self) -> str | None:
            raise NotImplementedError

    def __init__(
        self,
        conf: Configuration,
        resources: ResourceManagerImpl | None = None
    ) -> None:
        """使用提供的配置创建MemMachine实例"""
        self._conf = conf
        if resources is not None:
            self._resources = resources
        else:
            self._resources = ResourceManagerImpl(conf)

    async def start(self) -> None:
        """启动MemMachine服务"""
        semantic_service = await self._resources.get_semantic_service()
        await semantic_service.start()

    async def stop(self) -> None:
        """停止MemMachine服务"""
        semantic_service = await self._resources.get_semantic_service()
        await semantic_service.stop()
        await self._resources.close()
```

### 核心方法实现

#### 添加对话条目

```python
async def add_episodes(
    self,
    session_data: InstanceOf[SessionData],
    episode_entries: list[EpisodeEntry],
    *,
    target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
) -> list[EpisodeIdT]:
    """添加对话条目到指定记忆类型"""

    # 1. 存储对话条目到EpisodeStore
    episode_storage = await self._resources.get_episode_storage()
    episodes = await episode_storage.add_episodes(
        session_data.session_key,
        episode_entries,
    )
    episode_ids = [e.uid for e in episodes]

    tasks = []

    # 2. 添加到情景记忆
    if MemoryType.Episodic in target_memories:
        episodic_memory_manager = (
            await self._resources.get_episodic_memory_manager()
        )
        async with episodic_memory_manager.open_or_create_episodic_memory(
            session_key=session_data.session_key,
            description="",
            episodic_memory_config=self._with_default_episodic_memory_conf(
                session_key=session_data.session_key
            ),
            metadata={},
        ) as episodic_session:
            tasks.append(episodic_session.add_memory_episodes(episodes))

    # 3. 添加到语义记忆处理队列
    if MemoryType.Semantic in target_memories:
        semantic_session_manager = (
            await self._resources.get_semantic_session_manager()
        )
        tasks.append(
            semantic_session_manager.add_message(
                episode_ids=episode_ids,
                session_data=session_data,
            )
        )

    # 4. 并发执行
    await asyncio.gather(*tasks)
    return episode_ids
```

#### 跨记忆类型搜索

```python
class SearchResponse(BaseModel):
    """跨记忆类型的聚合搜索结果"""
    episodic_memory: EpisodicMemory.QueryResponse | None = None
    semantic_memory: list[SemanticFeature] | None = None

async def query_search(
    self,
    session_data: InstanceOf[SessionData],
    *,
    target_memories: list[MemoryType] = ALL_MEMORY_TYPES,
    query: str,
    limit: int | None = None,
    search_filter: str | None = None,
) -> SearchResponse:
    """跨记忆类型查询搜索"""

    episodic_task: Task | None = None
    semantic_task: Task | None = None

    property_filter = parse_filter(search_filter) if search_filter else None

    # 并发搜索不同记忆类型
    if MemoryType.Episodic in target_memories:
        episodic_task = asyncio.create_task(
            self._search_episodic_memory(
                session_data=session_data,
                query=query,
                limit=limit,
                search_filter=property_filter,
            )
        )

    if MemoryType.Semantic in target_memories:
        semantic_session = await self._resources.get_semantic_session_manager()
        semantic_task = asyncio.create_task(
            semantic_session.search(
                memory_type=[IsolationType.SESSION],
                message=query,
                session_data=session_data,
                limit=limit,
                search_filter=property_filter,
            )
        )

    return MemMachine.SearchResponse(
        episodic_memory=await episodic_task if episodic_task else None,
        semantic_memory=await semantic_task if semantic_task else None,
    )
```

## 资源管理架构

### ResourceManagerImpl

资源管理器负责管理所有依赖项的创建和生命周期：

```python
class ResourceManagerImpl:
    """资源管理器实现"""

    def __init__(self, conf: Configuration):
        self._conf = conf
        self._services = {}  # 缓存服务实例

    async def get_episode_storage(self) -> EpisodeStorage:
        """获取对话存储服务"""
        if 'episode_storage' not in self._services:
            # 根据配置创建存储服务
            storage = self._create_episode_storage()
            await storage.startup()
            self._services['episode_storage'] = storage
        return self._services['episode_storage']

    async def get_semantic_service(self) -> SemanticService:
        """获取语义记忆服务"""
        if 'semantic_service' not in self._services:
            service = SemanticService(
                params=SemanticService.Params(
                    semantic_storage=await self.get_semantic_storage(),
                    episode_storage=await self.get_episode_storage(),
                    resource_retriever=self,
                    consolidation_threshold=20,
                    debug_fail_loudly=False,
                )
            )
            self._services['semantic_service'] = service
        return self._services['semantic_service']
```

## 配置管理系统

### Configuration类

基于源码分析，配置系统支持：

```python
class Configuration:
    """MemMachine配置管理"""

    def __init__(self):
        # 默认配置
        self.default_long_term_memory_embedder = "text-embedding-ada-002"
        self.default_long_term_memory_reranker = "default-reranker"

        # 情景记忆配置
        self.episodic_memory = EpisodicMemoryConfig()

        # 语义记忆配置
        self.semantic_memory = SemanticMemoryConfig()

    def check_embedder(self, embedder_name: str) -> None:
        """验证嵌入器配置"""
        if embedder_name not in self.available_embedders:
            raise ValueError(f"Unknown embedder: {embedder_name}")

    def check_reranker(self, reranker_name: str) -> None:
        """验证重排序器配置"""
        if reranker_name not in self.available_rerankers:
            raise ValueError(f"Unknown reranker: {reranker_name}")
```

## 记忆类型架构

### 双重记忆系统设计

MemMemory采用双重记忆系统架构：

#### 1. 情景记忆（Episodic Memory）
- **作用**：存储具体的对话事件和经历
- **特点**：按时间序列组织，支持上下文检索
- **结构**：
  - 短期记忆（ShortTermMemory）：当前会话的临时信息
  - 长期记忆（LongTermMemory）：持久化的会话历史

#### 2. 语义记忆（Semantic Memory）
- **作用**：提取和存储语义特征
- **特点**：支持语义搜索和检索
- **功能**：
  - 特征提取和存储
  - 向量嵌入管理
  - 语义相似度搜索

### 隔离类型

基于`src/memmachine/semantic_memory/semantic_session_manager.py`：

```python
class IsolationType(Enum):
    """记忆隔离类型"""
    USER = "user_profile"      # 用户级隔离
    ROLE = "role_profile"      # 角色级隔离
    SESSION = "session"        # 会话级隔离
```

## 异步架构设计

### 并发处理模式

MemMachine采用完全异步的架构设计：

```python
# 并发写入多个记忆系统
tasks: list[Coroutine] = []
if self._short_term_memory:
    tasks.append(self._short_term_memory.add_episodes(episodes))
if self._long_term_memory:
    tasks.append(self._long_term_memory.add_episodes(episodes))
await asyncio.gather(*tasks)

# 并发搜索多个记忆类型
episodic_task = asyncio.create_task(search_episodic_memory(query))
semantic_task = asyncio.create_task(search_semantic_memory(query))
results = await asyncio.gather(episodic_task, semantic_task)
```

### 资源生命周期管理

```python
async def start(self) -> None:
    """启动所有服务"""
    await self._resource_manager.start()

async def stop(self) -> None:
    """停止所有服务"""
    await self._resource_manager.stop()
```

## 协议导向设计

MemMachine大量使用Python的Protocol进行接口定义：

```python
@runtime_checkable
class SessionData(Protocol):
    """会话数据协议"""
    @property
    def session_key(self) -> str: ...
    @property
    def user_profile_id(self) -> str | None: ...
    @property
    def role_profile_id(self) -> str | None: ...
    @property
    def session_id(self) -> str | None: ...

@runtime_checkable
class ResourceRetriever(Protocol):
    """资源检索器协议"""
    def get_resources(self, set_id: SetIdT) -> Resources: ...
```

## 性能监控集成

### 指标工厂

```python
from memmachine.common.metrics_factory import MetricsFactory

# 在EpisodicMemory中使用
self._ingestion_latency_summary = metrics_manager.get_summary(
    "Ingestion_latency",
    "Latency of Episode ingestion in milliseconds",
)
self._query_latency_summary = metrics_manager.get_summary(
    "query_latency",
    "Latency of query processing in milliseconds",
)
```

## 总结

MemMachine的架构设计展现了以下关键特征：

1. **模块化设计**：清晰的模块分离和职责划分
2. **异步架构**：全面采用asyncio实现高性能并发处理
3. **协议导向**：使用Protocol定义接口，提高灵活性
4. **双重记忆**：情景记忆和语义记忆的协同工作
5. **资源管理**：统一的资源管理和生命周期控制
6. **类型安全**：使用Pydantic确保数据模型的类型安全
7. **配置驱动**：灵活的配置管理系统
8. **性能监控**：内置Prometheus指标收集

这种架构设计使得MemMachine能够有效地为AI智能体提供多层次、高性能的记忆服务。