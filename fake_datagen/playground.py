from typing import Tuple
import pyarrow as pa  # type: ignore
from pyarrow import Field as PyarrowField
import pyarrow.compute as pc  # type: ignore


def assign_field_ids(pa_fields: list[PyarrowField], field_id: int = 0) -> Tuple[list[PyarrowField], int]:
    """Assign field ids to the schema."""
    new_fields = []
    for field in pa_fields:
        if isinstance(field.type, pa.StructType):
            field_indices = list(range(field.type.num_fields))
            struct_fields = [field.type.field(field_i) for field_i in field_indices]
            nested_pa_fields, field_id = assign_field_ids(struct_fields, field_id)
            new_fields.append(
                pa.field(field.name, pa.struct(nested_pa_fields), nullable=field.nullable, metadata=field.metadata)
            )
        else:
            field_id += 1
            field_with_metadata = field.with_metadata({"PARQUET:field_id": f"{field_id}"})
            new_fields.append(field_with_metadata)

    return new_fields, field_id


def main():
    pa_schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("age", pa.int32()),
            pa.field(
                "address",
                pa.struct(
                    [
                        pa.field("city", pa.string(), nullable=True),
                        pa.field("country", pa.string(), nullable=False),
                    ]
                ),
            ),
        ]
    )
    pa_fields_with_field_ids, _ = assign_field_ids(pa_schema)
    print(f"pa_fields_with_field_ids: {pa_fields_with_field_ids}")
    
    for field in pa_fields_with_field_ids:
        print(f"field.metadata: {field.metadata}")

    new_schema = pa.schema(pa_fields_with_field_ids)
    print(f"new_schema: {new_schema}")


if __name__ == "__main__":
    main()
