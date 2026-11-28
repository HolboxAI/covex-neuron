"""Host operands for a diagnostic replay across the 8K KV boundary (dense MRv2)."""

import hashlib


def page_record(raw_row, valid_pages, block_size, position, write_slot):
    logical_page, in_page = divmod(position, block_size)
    populated = raw_row[:valid_pages]
    physical = int(populated[logical_page]) if logical_page < len(populated) else None
    expected = physical * block_size + in_page if physical is not None else None
    return {
        "position": position, "block_size": block_size, "logical_page": logical_page,
        "physical_page": physical, "page_offset": in_page, "valid_pages": valid_pages,
        "observed_write_slot": write_slot, "expected_write_slot": expected,
        "slot_matches": expected == write_slot and expected is not None,
        "page_ids_sha256": hashlib.sha256(populated.astype("<i4", copy=False).tobytes()).hexdigest(),
        "adjacent_pages": {str(i): int(populated[i]) for i in range(max(logical_page - 1, 0),
                                                                    min(logical_page + 2, len(populated)))},
    }
