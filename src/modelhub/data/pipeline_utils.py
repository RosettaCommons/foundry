from datahub.transforms._checks import check_atom_array_annotation
from datahub.transforms.crop import compute_local_hash


def annotate_pre_crop_hash(data: dict) -> dict:
    hash_pre = compute_local_hash(data["atom_array"])
    data["atom_array"].set_annotation("hash_pre", hash_pre)
    return data


def annotate_post_crop_hash(data: dict) -> dict:
    hash_post = compute_local_hash(data["atom_array"])
    data["atom_array"].set_annotation("hash_post", hash_post)
    return data


def set_to_occupancy_0_where_crop_hashes_differ(data: dict) -> dict:
    check_atom_array_annotation(
        data["atom_array"], ["hash_pre", "hash_post", "occupancy"]
    )

    # Create a mask of where hash_pre != hash_post
    atom_array = data["atom_array"]
    mask = atom_array.get_annotation("hash_pre") != atom_array.get_annotation(
        "hash_post"
    )

    # Where the hashes differ, set occupancy to 0
    atom_array.occupancy[mask] = 0

    return data
