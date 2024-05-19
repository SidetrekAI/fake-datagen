import time
import pyarrow as pa  # type: ignore
from generators import gen_rand_dataset  # type: ignore
from examples.examples import get_ex_tables  # type: ignore


"""
poetry run python fake_datagen/main.py
"""

def main():
    start = time.time()
    ex_tables = get_ex_tables(num_records=100000)

    gen_rand_dataset(ex_tables, target="csv", target_config={"outdir": "data"})
    
    end = time.time()
    print(f"Completed in: {end - start}s")

if __name__ == "__main__":
    main()
