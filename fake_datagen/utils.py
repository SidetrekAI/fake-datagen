from typing import get_args
import numpy as np  # type: ignore
import pyarrow as pa  # type: ignore
import duckdb 
from mimesis import Generic
from mimesis.locales import Locale
from type_defs import Schema, FieldType, Table, Target  # type: ignore


mim = Generic(locale=Locale.EN)


type_to_pa_type = {
    "id": pa.uint32(),
    "int": pa.int32(),
    "float": pa.float32(),
    "str": pa.string(),
    "bool": pa.bool_(),
    "timestamp": pa.timestamp("ms"),
    "category": pa.string(),
    "mimesis": pa.string(),
}


def schema_to_pa_schema(schema: Schema) -> pa.Schema:
    pa_fields = []
    for field in schema:
        pa_fields.append(pa.field(field["name"], type_to_pa_type[field["type"]]))
    return pa.schema(pa_fields)


duckdb_conn = duckdb.connect()


def is_valid_field_type(s: str) -> bool:
    return s in get_args(FieldType)


def is_valid_target(t: str) -> bool:
    return t in get_args(Target)


def minmax_scaler(samples: np.ndarray, low: int | float, high: int | float) -> np.ndarray:
    return (samples - samples.min()) / (samples.max() - samples.min()) * (high - low) + low
