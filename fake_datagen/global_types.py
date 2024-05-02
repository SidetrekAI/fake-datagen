from typing import TypedDict, Optional, Literal, Callable


class Fkey(TypedDict):
    table: str
    id_field: str


FieldType = Literal[
    "id",
    "int",
    "float",
    "str",
    "bool",
    "timestamp",
    "category",
    "mimesis",
]


class Field(TypedDict):
    name: str
    type: str  # maps to pyarrow types
    start: Optional[int]  # for id data; defaults to 0
    str_len: Optional[int]  # for string data; defaults to 10
    categories: list  # for categorical data
    low: Optional[int | float]  # for numeric data (includes low, excludes high)
    high: Optional[int | float]  # for numeric data (includes low, excludes high)
    max: Optional[int]  # for numeric data
    p: Optional[list]  # probabiliy distribution for categorical data
    fkey: Optional[Fkey]
    mimesis_provider_fn: Optional[Callable] # mimesis provider function for mimesis type data - e.g. `mim.address.address`


class Schema(TypedDict):
    fields: list[Field]


class Table(TypedDict):
    name: str
    num_records: int
    schema: Schema


class ParquetTargetConfig(TypedDict):
    filepath: str


class CSVTargetConfig(TypedDict):
    filepath: str


class IcebergTargetConfig(TypedDict):
    s3_bucket: str
    rest_catalog_uri: str
    rest_catalog_name: str
    rest_catalog_namespace_name: str


Target = Literal["parquet", "csv", "iceberg"]
TargetConfig = ParquetTargetConfig | CSVTargetConfig | IcebergTargetConfig
