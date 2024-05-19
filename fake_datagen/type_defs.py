from typing import TypedDict, Optional, Literal, Callable, Union


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
    mimesis_provider_fn: Optional[
        Callable
    ]  # mimesis provider function for mimesis type data - e.g. `mim.address.address`


class Schema(TypedDict):
    fields: list[Field]


class Table(TypedDict):
    name: str
    num_records: int
    schema: Schema


class ParquetTargetConfig(TypedDict):
    outdir: str


class CSVTargetConfig(TypedDict):
    outdir: str


class IcebergTargetConfig(TypedDict):
    outdir: str
    s3_bucket: str
    rest_catalog_uri: str
    rest_catalog_name: str
    rest_catalog_namespace_name: str


Target = Literal["parquet", "csv", "iceberg"]
TargetConfig = ParquetTargetConfig | CSVTargetConfig | IcebergTargetConfig


# Distribution types


class NormalDist(TypedDict):
    type: Literal["normal"]
    loc: Optional[float]
    scale: Optional[float]


class BetaDist(TypedDict):
    type: Literal["beta"]
    a: Optional[float]
    b: Optional[float]


class GeometricDist(TypedDict):
    type: Literal["geometric"]
    prob: Optional[float]


class ExponentialDist(TypedDict):
    type: Literal["exponential"]
    scale: Optional[float]


class PowerDist(TypedDict):
    type: Literal["power"]
    a: float | list[float]


class UniformDist(TypedDict):
    type: Literal["uniform"]
    low: Optional[float]
    high: Optional[float]


class PoissonDist(TypedDict):
    type: Literal["poisson"]
    lam: Optional[float]


class LogNormalDist(TypedDict):
    type: Literal["lognormal"]
    mean: Optional[float]
    sigma: Optional[float]


Dist = Union[NormalDist, BetaDist, GeometricDist, ExponentialDist, PowerDist, UniformDist, PoissonDist, LogNormalDist]
