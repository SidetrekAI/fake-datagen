from global_types import Table  # type: ignore
from utils import mim  # type: ignore


gender = ["male", "female"]


def get_ex_tables(num_records: int) -> list[Table]:
    return [
        {
            "name": "orders",
            "num_records": num_records,
            "schema": [
                {"name": "order_id", "type": "id"},
                {"name": "created_at", "type": "timestamp", "start": "2022-01-01", "range_in_days": 365 * 2},
                {"name": "qty", "type": "int", "low": 1, "high": 10},
                {"name": "product_id", "type": "id", "fkey": {"table": "products", "id_field": "id"}},
                {"name": "customer_id", "type": "id", "fkey": {"table": "customers", "id_field": "id"}},
                {"name": "store_id", "type": "id", "fkey": {"table": "stores", "id_field": "id"}},
            ],
        },
        {
            "name": "customers",
            "num_records": int(num_records / 5),
            "schema": [
                {"name": "id", "type": "id"},
                {"name": "created_at", "type": "timestamp", "start": "2022-01-01", "range_in_days": 365 * 2},
                {"name": "first_name", "type": "mimesis", "mimesis_provider_fn": mim.person.first_name},
                {"name": "last_name", "type": "mimesis", "mimesis_provider_fn": mim.person.last_name},
                {"name": "gender", "type": "category", "categories": ["male", "female"]},
                {"name": "country", "type": "mimesis", "mimesis_provider_fn": mim.address.country},
                {"name": "address", "type": "mimesis", "mimesis_provider_fn": mim.address.address},
                {"name": "phone", "type": "mimesis", "mimesis_provider_fn": mim.person.telephone},
                {"name": "email", "type": "mimesis", "mimesis_provider_fn": mim.person.email},
                {
                    "name": "payment_method",
                    "type": "category",
                    "categories": ["credit_card", "debit_card", "paypal", "cash"],
                },
                {
                    "name": "traffic_source",
                    "type": "category",
                    "categories": ["Search", "Direct", "Email", "Social", "PPC"],
                },
                {
                    "name": "referrer",
                    "type": "mimesis",
                    "mimesis_provider_fn": mim.internet.hostname,
                },
                {"name": "customer_age", "type": "int", "low": 18, "high": 80},
                {"name": "device_type", "type": "category", "categories": ["desktop", "mobile"], "p": [0.3, 0.7]},
            ],
        },
        {
            "name": "products",
            "num_records": int(num_records / 50),
            "schema": [
                {"name": "id", "type": "id"},
                {"name": "name", "type": "mimesis", "mimesis_provider_fn": mim.food.dish},
                {
                    "name": "category",
                    "type": "category",
                    "categories": [
                        "Electronics",
                        "Home & Garden",
                        "Fashion",
                        "Books",
                        "Video Games",
                        "Health & Hygine",
                        "Music Instruments",
                        "Beauty & Personal",
                        "Sports & Outdoor",
                        "Office Supply",
                    ],
                },
                {"name": "price", "type": "int", "low": 500, "high": 20000},
                {"name": "description", "type": "mimesis", "mimesis_provider_fn": mim.text.title},
            ],
        },
        {
            "name": "stores",
            "num_records": int(num_records / 500),
            "schema": [
                {"name": "id", "type": "id"},
                {"name": "name", "type": "mimesis", "mimesis_provider_fn": mim.text.word},
                {"name": "city", "type": "mimesis", "mimesis_provider_fn": mim.address.city},
                {"name": "state", "type": "mimesis", "mimesis_provider_fn": mim.address.state},
                {"name": "tax_rate", "type": "float", "low": 0.05, "high": 0.15},
            ],
        },
    ]
