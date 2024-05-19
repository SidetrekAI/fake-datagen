import timeit
import numpy as np
import matplotlib.pyplot as plt
from generators import (  # type: ignore
    gen_ids,
    gen_rand_strs,
    gen_rand_float,
    gen_rand_integers,
    gen_rand_categories,
    gen_rand_timestamps,
    gen_mimesis_data,
    gen_rand_pa_table,
    gen_rand_pa_tables,
)
from examples.examples import get_ex_tables, gender  # type: ignore
from utils import mim  # type: ignore


str_len = 10
num_records = 1000
loop = 10

ex_tables = get_ex_tables(num_records)

print("num_records=", "{:,}".format(num_records))
print("test_loops=", "{:,}".format(loop))
print("total_iterations=", "{:,}".format(num_records * loop))
print("")

# print("gen_ids=", timeit.timeit(stmt="gen_ids(num_records)", globals=globals(), number=loop))
# print("gen_rand_strs=", timeit.timeit(stmt="gen_rand_strs(num_records, str_len)", globals=globals(), number=loop))
# print("gen_rand_float=", timeit.timeit(stmt="gen_rand_float(0, 1, num_records)", globals=globals(), number=loop))
# print(
#     "gen_rand_integers=", timeit.timeit(stmt="gen_rand_integers(0, 1000, num_records)", globals=globals(), number=loop)
# )
# print(
#     "gen_rand_integers_dist=",
#     timeit.timeit(
#         stmt="gen_rand_integers(0, 1000, num_records, dist={'type': 'normal'})", globals=globals(), number=loop
#     ),
# )
# plt.hist(gen_rand_integers(0, 1000, num_records))
# plt.hist(gen_rand_integers(0, 1000, num_records, dist={"type": "power", "a": 1.1}))
# plt.show()
# print(
#     "gen_rand_categories=",
#     timeit.timeit(
#         stmt="gen_rand_categories(gender, num_records)",
#         globals=globals(),
#         number=loop,
#     ),
# )
# print(
#     "gen_rand_timestamps=",
#     timeit.timeit(
#         stmt="gen_rand_timestamps(np.datetime64('2021-01-01'), 1000, num_records)",
#         globals=globals(),
#         number=loop,
#     ),
# )
# print(
#     "gen_rand_timestamps=",
#     timeit.timeit(
#         stmt="gen_rand_timestamps(np.datetime64('2021-01-01'), 1000, num_records, dist={'type': 'normal'})",
#         globals=globals(),
#         number=loop,
#     ),
# )
# plt.hist(gen_rand_timestamps(np.datetime64("2021-01-01"), 1000, num_records, dist={"type": "power", "a": 1.1}))
# plt.show()
# print(
#     "gen_mimesis_data=",
#     timeit.timeit(stmt="gen_mimesis_data(num_records, mim.address.address)", globals=globals(), number=loop),
# )
# print(
#     "gen_rand_pa_table=",
#     timeit.timeit(
#         stmt='gen_rand_pa_table(ex_tables[0]["name"], ex_tables)',
#         globals=globals(),
#         number=loop,
#     ),
# )
# print("gen_rand_pa_tables=", timeit.timeit(stmt="gen_rand_pa_tables(ex_tables)", globals=globals(), number=loop))
