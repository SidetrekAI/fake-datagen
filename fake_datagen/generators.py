from typing import Optional, Callable, List
import numpy as np # type: ignore
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pyarrow.csv as pcsv  # type: ignore
from type_defs import Table, Target, TargetConfig, Dist  # type: ignore
from utils import schema_to_pa_schema, is_valid_field_type, is_valid_target, minmax_scaler  # type: ignore


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
def gen_rand_integers(low: int, high: int, size: int, dist: Dist = None) -> np.ndarray:
    if dist is None:
        return rng.integers(low=low, high=high, size=size)
    
    dist_type = dist["type"]

    if dist_type == "normal":
        samples = rng.normal(loc=0, scale=1, size=size)
        scaled_samples = minmax_scaler(samples, low, high)
        return np.clip(scaled_samples, low, high).astype(int)
    if dist_type == "power":
        samples = rng.power(dist["a"], size=size)
        scaled_samples = minmax_scaler(samples, low, high)
        return np.clip(scaled_samples, low, high).astype(int)
    else:
        raise ValueError(f"Unsupported distribution type in gen_rand_integers: {dist_type}")


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
    categories: list[str], 
    size: int,
    p: list[float] | None = None, 
    dist: Dist = None
) -> np.ndarray:
    np_categories = np.array(categories)
    num_categories = len(categories)
    
    if dist is not None:
        dist_type = dist["type"]

        if dist_type == "normal":
            loc = dist.get("loc", 0.5)
            scale = dist.get("scale", 0.1)
            samples = rng.normal(loc=loc, scale=scale, size=size)
            samples = (samples - samples.min()) / (samples.max() - samples.min())  # Normalize to [0, 1]
            indices = (samples * num_categories).astype(int)
            indices = np.clip(indices, 0, num_categories - 1)
        
        elif dist_type == "beta":
            a = dist.get("a", 2.0)
            b = dist.get("b", 5.0)
            samples = rng.beta(a=a, b=b, size=size)
            indices = (samples * num_categories).astype(int)
            indices = np.clip(indices, 0, num_categories - 1)
        
        elif dist_type == "geometric":
            prob = dist.get("prob", 0.5)
            indices = rng.geometric(p=prob, size=size) - 1
            indices = np.clip(indices, 0, num_categories - 1)
        
        elif dist_type == "exponential":
            scale = dist.get("scale", 1.0)
            samples = rng.exponential(scale=scale, size=size)
            samples = (samples - samples.min()) / (samples.max() - samples.min())  # Normalize to [0, 1]
            indices = (samples * num_categories).astype(int)
            indices = np.clip(indices, 0, num_categories - 1)
        
        elif dist_type == "uniform":
            low = dist.get("low", 0.0)
            high = dist.get("high", 1.0)
            samples = rng.uniform(low=low, high=high, size=size)
            
            # Normalize samples to [0, 1]
            normalized_samples = (samples - low) / (high - low)
            
            # Scale to range [0, num_categories - 1]
            indices = (normalized_samples * num_categories).astype(int)
            
            # Ensure indices are within valid range
            indices = np.clip(indices, 0, num_categories - 1)
        
        elif dist_type == "poisson":
            lam = dist.get("lam", 1.0)
            indices = rng.poisson(lam=lam, size=size)
            indices = np.clip(indices, 0, num_categories - 1)
        
        elif dist_type == "lognormal":
            mean = dist.get("mean", 0.0)
            sigma = dist.get("sigma", 1.0)
            samples = rng.lognormal(mean=mean, sigma=sigma, size=size)
            samples = (samples - samples.min()) / (samples.max() - samples.min())  # Normalize to [0, 1]
            indices = (samples * num_categories).astype(int)
            indices = np.clip(indices, 0, num_categories - 1)
        
        else:
            raise ValueError(f"Unsupported distribution type in gen_rand_categories: {dist_type}")
    else:
        indices = rng.choice(len(categories), size=size, p=p)

    return np_categories[indices]


def gen_rand_timestamps(
    start: np.datetime64, 
    range_in_days: int, 
    size: int, 
    dtype: str = "datetime64[ms]", 
    dist: Dist = None
) -> np.ndarray:
    start = np.datetime64(start)
    base = np.full(size, start)
    
    if dist is None:
        offset = rng.integers(0, range_in_days, size)
        offset = offset.astype("timedelta64[D]")
        return (base + offset).astype(dtype)

    dist_type = dist["type"]

    if dist_type == "normal":
        samples = rng.normal(loc=0, scale=1, size=size)
        offset = minmax_scaler(samples, low=0, high=range_in_days).astype("timedelta64[D]")
        # offset = np.interp(samples, (samples.min(), samples.max()), (0, range_in_days)).astype("timedelta64[D]")  # This is slightly faster!
        return (base + offset).astype(dtype)
    if dist_type == "power":
        samples = rng.power(dist["a"], size=size)
        offset = minmax_scaler(samples, low=0, high=range_in_days).astype("timedelta64[D]")
        # offset = np.interp(samples, (samples.min(), samples.max()), (0, range_in_days)).astype("timedelta64[D]")  # This is slightly faster!
        return (base + offset).astype(dtype)
    else:
        raise ValueError(f"Unsupported distribution type in gen_rand_integers: {dist_type}")


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
            """
            `fkey` fields require special treatment.
                
            Since ids are always incremental, we can infer the low and high value for generating
            random integers for the fkey field.
            """
            fkey_table = next(
                (t for t in tables if t["name"] == field["fkey"]["table"]), None
            )
            
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
            records.append(pa.array(gen_rand_integers(low=low, high=high, size=num_records, dist=field.get("dist", None))))
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
                records.append(pa.array(gen_rand_integers(field["low"], field["high"], num_records, dist=field.get("dist", None))))

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
                records.append(pa.array(gen_rand_timestamps(field["start"], field["range_in_days"], num_records, dist=field.get("dist", None))))

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
