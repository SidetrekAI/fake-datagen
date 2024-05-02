import time
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from generators import gen_mimesis_data, gen_rand_pa_tables, gen_rand_dataset  # type: ignore
from examples.examples import get_ex_tables  # type: ignore
from utils import duckdb_conn, mim  # type: ignore

start = time.time()
ex_tables = get_ex_tables(num_records=100000)

# pa_tables = gen_rand_pa_tables(ex_tables)
# pa_sample_table = pa_tables[0]
# results = duckdb_conn.execute("select * from pa_sample_table limit 10").arrow()
# print(f"pa_sample_table={results}")
# pa_sample_table2 = pa_tables[1]
# results2 = duckdb_conn.execute("select * from pa_sample_table2 limit 10").arrow()
# print(f"pa_sample_table2={results2}")

gen_rand_dataset(ex_tables, target="csv", target_config={"filepath": "data/csv"})

# result = gen_mimesis_data(100, mim.text.word)
# print(result)

end = time.time()
print(f"Completed in: {end - start}s")
