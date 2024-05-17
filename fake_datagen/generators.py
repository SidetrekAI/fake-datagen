from typing import Optional, Callable
import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pyarrow.csv as pcsv  # type: ignore
from global_types import Table, Target, TargetConfig  # type: ignore
from utils import schema_to_pa_schema, is_valid_field_type, is_valid_target  # type: ignore


"""
Generators for generating fake data

NOTE: The ones with `WARNING: SLOW!!!` should never be used for performance testing fake data generation.
      Only use these for generating small datasets for demo purposes.
"""

rng = np.random.default_rng(seed=42)


# -----------------------------------------------------
# FAST GENERATORS (for performance testing)
# -----------------------------------------------------

# Generates incremental ids
def gen_ids(size: int, start: int = 0) -> np.ndarray:
    return np.arange(start, size)


# Generates `size` integers with uniform distribution
def gen_rand_integers(low: int, high: int, size: int) -> np.ndarray:
    return rng.integers(low=low, high=high, size=size)


# Generates `size` float with uniform distribution
def gen_rand_float(low: int | float, high: int | float, size: int) -> np.ndarray:
    return rng.uniform(low=low, high=high, size=size)


# Generates `size` strings (lowercase alpahabet - i.e. `a-z``) with uniform distribution
def gen_rand_strs(size: int, str_len: int = 10) -> np.ndarray:
    a, z = np.array(["a", "z"]).view("int32")
    return rng.integers(low=a, high=z, size=size * str_len, dtype="int32").view(
        f"U{str_len}"
    )
    

# Generates `size` categories with various distributions
def gen_rand_categories(
    categories: list, size: int, p: list[float] | None = None, dist: str | None = None
) -> np.ndarray:
    np_categories = np.array(categories)
    if dist is not None:
        if dist == "normal":
            loc = p[0] if p and len(p) > 0 else 0.5
            scale = p[1] if p and len(p) > 1 else 0.1
            values = rng.normal(loc=loc, scale=scale, size=size)
            indices = np.digitize(values, bins=np.linspace(0, 1, len(categories) + 1)) - 1
        elif dist == "beta":
            a = p[0] if p and len(p) > 0 else 2.0
            b = p[1] if p and len(p) > 1 else 5.0
            values = rng.beta(a=a, b=b, size=size)
            indices = np.digitize(values, bins=np.linspace(0, 1, len(categories) + 1)) - 1
        elif dist == "geometric":
            prob = p[0] if p and len(p) > 0 else 0.5
            indices = rng.geometric(p=prob, size=size) - 1
            indices = np.clip(indices, 0, len(categories) - 1)
        elif dist == "exponential":
            scale = p[0] if p and len(p) > 0 else 1.0
            values = rng.exponential(scale=scale, size=size)
            indices = np.digitize(values, bins=np.linspace(0, max(values), len(categories))) - 1
        elif dist == "uniform":
            low = p[0] if p and len(p) > 0 else 0.0
            high = p[1] if p and len(p) > 1 else 1.0
            values = rng.uniform(low=low, high=high, size=size)
            indices = np.digitize(values, bins=np.linspace(low, high, len(categories) + 1)) - 1
        elif dist == "poisson":
            lam = p[0] if p and len(p) > 0 else 1.0
            indices = rng.poisson(lam=lam, size=size)
            indices = np.clip(indices, 0, len(categories) - 1)
        elif dist == "lognormal":
            mean = p[0] if p and len(p) > 0 else 0.0
            sigma = p[1] if p and len(p) > 1 else 1.0
            values = rng.lognormal(mean=mean, sigma=sigma, size=size)
            values = (values - values.min()) / (values.max() - values.min())  # Normalize to [0, 1]
            indices = np.digitize(values, bins=np.linspace(0, 1, len(categories) + 1)) - 1
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")
    else:
        indices = rng.choice(len(categories), size=size, p=p)

    return np_categories[indices]

def gen_rand_timestamps(
    start: np.datetime64, range_in_days: int, size: int, dtype: str = "datetime64[ms]"
) -> np.ndarray:
    start = np.datetime64(start)
    base = np.full(size, start)
    offset = rng.integers(0, range_in_days, size)
    offset = offset.astype("timedelta64[D]")
    return (base + offset).astype("datetime64[ms]")



# -----------------------------------------------------
# SLOW GENERATORS (for demo purposes only)
# -----------------------------------------------------


# -----------------------------------------------------
# MIMESIS GENERATORS - SLOW!!! (for demo purposes only)
# -----------------------------------------------------

def gen_mimesis_data(size: int, mimesis_provider_fn: Callable) -> list:
    return [mimesis_provider_fn() for _ in range(size)]


# -----------------------------------------------------
# Table generators
# -----------------------------------------------------

def gen_rand_pa_table(table_name: str, tables: list[Table]) -> pa.Table:
    table = next((t for t in tables if t["name"] == table_name), None)
    
    if table is None:
        raise ValueError(f"Table {table_name} not found in the list of tables")
    
    schema = table["schema"]
    num_records = table["num_records"]
    records = []

    for field in schema:
        if not is_valid_field_type(field["type"]):
            raise ValueError(f'Invalid field type in schema: {(table["name"], field["name"])}')

    for field in schema:
        if "fkey" in field:
            fkey_table = next((t for t in tables if t["name"] == field["fkey"]["table"]), None)
            
            if fkey_table is None:
                raise ValueError(f"Table {field['fkey']['table']} not found in the list of tables")
            
            fkey_table_id_field = next(
                (f for f in fkey_table["schema"] if f["name"] == field["fkey"]["id_field"]),
                None,
            )
            
            if fkey_table_id_field is None:
                raise ValueError(f"Field {field['fkey']['id_field']} not found in the schema of {field['fkey']['table']}")
            
            low = int(fkey_table_id_field.get("start", 0))
            high = low + int(fkey_table["num_records"])
            records.append(pa.array(gen_rand_integers(low=low, high=high, size=num_records)))
        else:
            if field["type"] == "id":
                records.append(pa.array(gen_ids(num_records, field.get("start", 0))))

            elif field["type"] == "float":
                if "low" not in field or "high" not in field:
                    raise ValueError("low and high must be provided for float fields")
                records.append(pa.array(gen_rand_float(field["low"], field["high"], num_records)))

            elif field["type"] == "int":
                if "low" not in field or "high" not in field:
                    raise ValueError("low and high must be provided for int fields")
                records.append(pa.array(gen_rand_integers(field["low"], field["high"], num_records)))

            elif field["type"] == "str":
                records.append(pa.array(gen_rand_strs(num_records)))

            elif field["type"] == "category":
                if "categories" not in field:
                    raise ValueError("categories must be provided for category fields")
                records.append(
                    pa.array(gen_rand_categories(field["categories"], num_records, field.get("p", None), field.get("dist", None)))
                )

            elif field["type"] == "timestamp":
                if "start" not in field or "range_in_days" not in field:
                    raise ValueError("start and range_in_days must be provided for timestamp fields")
                records.append(pa.array(gen_rand_timestamps(field["start"], field["range_in_days"], num_records)))

            elif field["type"] == "mimesis":
                if "mimesis_provider_fn" not in field:
                    raise ValueError("mimesis_provider_fn must be provided for mimesis fields")
                records.append(pa.array(gen_mimesis_data(num_records, field["mimesis_provider_fn"])))

    pa_schema = schema_to_pa_schema(schema)
    return pa.table(records, schema=pa_schema)


def gen_rand_pa_tables(tables: list[Table]) -> list[pa.Table]:
    return [gen_rand_pa_table(table["name"], tables) for table in tables]


def gen_rand_dataset(tables: list[Table], target: Optional[Target] = None, target_config: Optional[TargetConfig] = None) -> None:
    if not is_valid_target(target):
        raise ValueError("Invalid target")

    for table in tables:
        pa_table = gen_rand_pa_table(table["name"], tables)
        outdir = target_config.get("outdir", "") if target_config else ""
        print('outdir:', outdir)

        if target == "parquet":
            filepath = f"{outdir}/{table['name']}.parquet"
            pq.write_table(pa_table, filepath)
        elif target == "csv":
            filepath = f"{outdir}/{table['name']}.csv"
            pcsv.write_csv(pa_table, filepath)
        elif target == "iceberg":
            raise NotImplementedError("Not yet implemented")
        else:
            raise ValueError("Invalid storage type")
