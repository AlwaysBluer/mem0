from typing import Optional, Any, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from mem0.llms.configs import LlmConfig


class Neo4jConfig(BaseModel):
    url: Optional[str] = Field(None, description="Host address for the graph database")
    username: Optional[str] = Field(None, description="Username for the graph database")
    password: Optional[str] = Field(None, description="Password for the graph database")
    database: Optional[str] = Field(None, description="Database for the graph database")
    base_label: Optional[bool] = Field(None, description="Whether to use base node label __Entity__ for all entities")

    @model_validator(mode="before")
    def check_host_port_or_path(cls, values):
        url, username, password = (
            values.get("url"),
            values.get("username"),
            values.get("password"),
        )
        if not url or not username or not password:
            raise ValueError("Please provide 'url', 'username' and 'password'.")
        return values


class MemgraphConfig(BaseModel):
    url: Optional[str] = Field(None, description="Host address for the graph database")
    username: Optional[str] = Field(None, description="Username for the graph database")
    password: Optional[str] = Field(None, description="Password for the graph database")

    @model_validator(mode="before")
    def check_host_port_or_path(cls, values):
        url, username, password = (
            values.get("url"),
            values.get("username"),
            values.get("password"),
        )
        if not url or not username or not password:
            raise ValueError("Please provide 'url', 'username' and 'password'.")
        return values


class GremlinGraphConfig(BaseModel):
    url: str = Field(description="Host address for the graph database")
    username: str = Field(description="Username for the graph database")
    password: str = Field(description="Password for the graph database")
    traversal_source: str = Field(description="Traversal source for the graph database", default="g")
    message_serializer: str = Field(description="Message serializer for the graph database", default=None)
    base_label: bool = Field(description="Whether to use base node label __Entity__ for all entities", default=True)

    @model_validator(mode="before")
    def check_host_port_or_path(cls, values):
        url, username, password = (
            values.get("url"),
            values.get("username"),
            values.get("password"),
        )
        if not url or not username or not password:
            raise ValueError("Please provide 'url', 'username' and 'password'.")
        return values


class GraphStoreConfig(BaseModel):
    provider: str = Field(description="Provider of the data store (e.g., 'neo4j')", default="neo4j")
    config: Union[Neo4jConfig, MemgraphConfig, GremlinGraphConfig] = Field(description="Configuration for the specific data store", default=None)
    llm: Optional[LlmConfig] = Field(description="LLM configuration for querying the graph store", default=None)
    custom_prompt: Optional[str] = Field(
        description="Custom prompt to fetch entities from the given text", default=None
    )

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider == "neo4j":
            return Neo4jConfig(**v.model_dump())
        elif provider == "memgraph":
            return MemgraphConfig(**v.model_dump())
        elif provider == "gremlin":
            return GremlinGraphConfig(**v.model_dump())
        else:
            raise ValueError(f"Unsupported graph store provider: {provider}")
