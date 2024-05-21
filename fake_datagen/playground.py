import os
from pathlib import Path
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/orders.csv")
    df = df.astype({"created_at": "datetime64[ms]"})
    print(df.head(10))
    df.hist(column="created_at")
    plt.show()


if __name__ == "__main__":
    main()
