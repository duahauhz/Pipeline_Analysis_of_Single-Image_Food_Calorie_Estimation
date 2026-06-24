"""Parse ECUSTFD density.xls → data/density_processed.json.

Source: ECUSTFD-calorie-estimation-using-food-image/density.xls (OLE Compound File).

Layout
------
- Workbook có 20 sheet; mỗi sheet đặt tên theo class (`apple`, `banana`, ..., `mix`).
- Mỗi sheet có header ở hàng 0: `id | type | volume(mm^3) | weight(g)`.
- Mỗi hàng dữ liệu tiếp theo là một đối tượng thực phẩm.
- Sheet `mix` chứa các ảnh trộn (mỗi ảnh có nhiều dòng, mỗi dòng một loại).
- Một số id xuất hiện hai lần (ảnh mix với 2 loại thức ăn), vd `mix001` xuất hiện
  ở cả dòng apple và orange.
- Cột `volume(mm^3)` thực tế chứa **cm³**: apple ~310 cm³ ↔ 244.5 g → density ~0.79
  g/cm³ (đúng theo bảng ECUSTFD). Nếu đúng mm³ thì apple chỉ nặng 0.31 g, vô lý.
  → Script ghi cả hai key `volume_cm3` và `volume_raw_mm3` để minh bạch.

Output JSON structure
---------------------
{
  "source_file": "<absolute path>",
  "n_sheets": 20,
  "classes": ["apple", "banana", ...],
  "per_image": {
      "<image_id>": {
          "class": "<class>",
          "volume_cm3": <float>,
          "volume_raw_mm3": <float>,
          "weight_g": <float>,
          "density_g_cm3": <float>
      },
      ...
  },
  "per_class": {
      "<class>": {
          "n": <int>,
          "mean_volume_cm3": <float>,
          "std_volume_cm3": <float>,
          "mean_weight_g": <float>,
          "std_weight_g": <float>,
          "mean_density_g_cm3": <float>,
          "kcal_per_100g_estimate": <float>  # computed from mean_weight → kcal via
                                            # class mean, only used as a hint when
                                            # the user's class is not in the prototype
      },
      ...
  }
}

CLI
---
    python scripts/parse_density_xls.py \\
        --xls data/density.xls \\
        --out data/density_processed.json

Defaults are project-relative.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import xlrd

EXPECTED_CLASSES = {
    "apple", "banana", "bread", "bun", "doughnut", "egg", "fired_dough_twist",
    "grape", "lemon", "litchi", "mango", "mooncake", "orange", "pear",
    "peach", "plum", "qiwi", "sachima", "tomato", "mix",
}


def _to_float(value) -> float:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Empty numeric cell")
        return float(value)
    return float(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std_sample(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def parse_workbook(xls_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return (per_image, per_class) from the workbook.

    The OLE Compound File reader (xlrd >= 2.0) keeps things simple here. We
    keep the original sheet name as authoritative for `class` (column 1 in
    each sheet is redundant with the sheet name, except for the `mix` sheet
    which contains rows for multiple classes).
    """
    if not xls_path.exists():
        raise FileNotFoundError(f"Missing density workbook: {xls_path}")
    book = xlrd.open_workbook(str(xls_path))

    per_image: dict[str, dict] = {}
    per_class_raw: dict[str, list[dict]] = defaultdict(list)

    for sheet in book.sheets():
        sheet_name = sheet.name.strip()
        if sheet.nrows < 2:
            continue

        # Verify the first row looks like the expected header.
        header = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
        if header != ["id", "type", "volume(mm^3)", "weight(g)"]:
            # Skip silently but note for the operator.
            print(f"  [warn] Unexpected header in sheet '{sheet_name}': {header}")

        for r in range(1, sheet.nrows):
            row = {
                "id": str(sheet.cell_value(r, 0)).strip(),
                "type": str(sheet.cell_value(r, 1)).strip(),
                "volume_raw_mm3": _to_float(sheet.cell_value(r, 2)),
                "weight_g": _to_float(sheet.cell_value(r, 3)),
            }
            if not row["id"]:
                continue

            # Class of this object: in `mix` sheet the type column tells us
            # which food is in the mixed image. For other sheets the type
            # column duplicates the sheet name, but we still trust it.
            obj_class = row["type"] or sheet_name

            # Treat the volume column as cm³ (see module docstring).
            row["volume_cm3"] = row["volume_raw_mm3"]
            if row["volume_cm3"] > 0:
                row["density_g_cm3"] = row["weight_g"] / row["volume_cm3"]
            else:
                row["density_g_cm3"] = 0.0
            row["class"] = obj_class

            # First occurrence of an id wins (multiple rows for `mix*` ids
            # all describe the same physical image from different angles).
            if row["id"] not in per_image:
                per_image[row["id"]] = {
                    "class": obj_class,
                    "classes": [obj_class],
                    "volume_cm3": row["volume_cm3"],
                    "volume_raw_mm3": row["volume_raw_mm3"],
                    "weight_g": row["weight_g"],
                    "density_g_cm3": row["density_g_cm3"],
                }
            else:
                # If the existing record is for a different class, expand it
                # to a multi-class record so both classes of a mix image are
                # preserved. `class` is set to the first class seen; `classes`
                # is always the full list.
                existing = per_image[row["id"]]
                if obj_class not in existing["classes"]:
                    existing["classes"].append(obj_class)
                    existing["volume_cm3"] = existing["volume_cm3"] + row["volume_cm3"]
                    existing["volume_raw_mm3"] = (
                        existing["volume_raw_mm3"] + row["volume_raw_mm3"]
                    )
                    existing["weight_g"] = existing["weight_g"] + row["weight_g"]
                    if existing["volume_cm3"] > 0:
                        existing["density_g_cm3"] = (
                            existing["weight_g"] / existing["volume_cm3"]
                        )

            per_class_raw[obj_class].append(
                {
                    "volume_cm3": row["volume_cm3"],
                    "weight_g": row["weight_g"],
                    "density_g_cm3": row["density_g_cm3"],
                }
            )

    per_class: dict[str, dict] = {}
    for cls, records in sorted(per_class_raw.items()):
        volumes = [r["volume_cm3"] for r in records]
        weights = [r["weight_g"] for r in records]
        densities = [r["density_g_cm3"] for r in records if r["density_g_cm3"] > 0]
        per_class[cls] = {
            "n": len(records),
            "mean_volume_cm3": _mean(volumes),
            "std_volume_cm3": _std_sample(volumes),
            "mean_weight_g": _mean(weights),
            "std_weight_g": _std_sample(weights),
            "mean_density_g_cm3": _mean(densities) if densities else 0.0,
            # kcal estimate is not provided in the workbook; we leave it None
            # and let the consumer fill it from a real nutrition source.
            "kcal_per_100g_external": None,
        }

    return per_image, per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xls",
        type=Path,
        default=Path("data/density.xls"),
        help="Path to the ECUSTFD density.xls workbook.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/density_processed.json"),
        help="Where to write the processed JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[parse_density_xls] Reading {args.xls}")
    per_image, per_class = parse_workbook(args.xls)

    classes = sorted(per_class.keys())
    # Note: the `mix` sheet is intentionally merged into individual classes
    # (its `type` column lists the actual food, not "mix"). We don't expect
    # `mix` to appear as a per-class key.
    missing = EXPECTED_CLASSES - set(classes) - {"mix"}
    if missing:
        print(f"  [warn] Expected classes not found: {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_file": str(args.xls.resolve()),
        "n_sheets": len(classes),
        "classes": classes,
        "per_image": per_image,
        "per_class": per_class,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[parse_density_xls] Wrote {args.out}")
    print(f"  Classes: {len(classes)}")
    print(f"  Per-image records: {len(per_image)}")
    print(f"  Per-class records: {len(per_class)}")
    # Show a small preview.
    sample_id = next(iter(per_image.keys()))
    print(f"  Example per_image[{sample_id!r}]: {per_image[sample_id]}")


if __name__ == "__main__":
    main()
