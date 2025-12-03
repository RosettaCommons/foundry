from typing import Any, Dict, List, Optional, Union

import numpy as np
from biotite.structure import AtomArray, get_residue_starts
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from foundry.utils.components import (
    ComponentStr,
    fetch_mask_from_idx,
    get_name_mask,
    split_contig,
    unravel_components,
)

# ============================================================================
# Input Specification & Validation
# ============================================================================


class InputSelection(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        str_min_length=1,
    )
    data: Dict[ComponentStr | str, List[str]] = Field(
        ..., description="Validated selection dictionary", exclude=True
    )
    raw: Any = Field(..., description="Original input value")
    mask: np.ndarray[np.bool_] = Field(
        ..., description="Boolean mask over atom array", exclude=True
    )
    tokens: Optional[Dict[ComponentStr | str, AtomArray]] = Field(
        ..., description="Selected atom arrays per component", exclude=True
    )

    @classmethod
    def from_any(
        cls, v: Union[str, bool, dict, None], atom_array: AtomArray
    ) -> Optional["InputSelection"]:
        """Create InputSelection from various input types."""
        if v is None:
            return None
        data, mask, _ = from_any_(v=v, atom_array=atom_array)
        return cls(
            raw=v,
            data=data,
            mask=mask,
            tokens=None,
        )

    @model_validator(mode="after")
    def check_keys(self):
        # lightweight validation that all keys have contig format (are splittable indices)
        assert all([split_contig(k) for k in self.data.keys()])
        return self

    # Wrapper functionality as dict-like
    def __getitem__(self, key: str) -> List[str]:
        """Allow dict-like access."""
        return self.data[key]

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def get(self, *args, **kwargs):
        return self.data.get(*args, **kwargs)

    # Serialization & repr
    def __repr__(self) -> str:
        num_atoms = self.mask.sum() if hasattr(self.mask, "sum") else 0
        num_tokens = len(self.tokens) if self.tokens else 0
        return (
            f"InputSelection(raw={self.raw!r}, atoms={num_atoms}, tokens={num_tokens})"
        )

    @model_serializer
    def serialize_raw(self) -> Any:
        return self.raw

    # Listed as separate methods for future changes to parsing.
    def get_mask(self):
        return self.mask

    def get_tokens(self, aa):
        _, _, tokens = from_any_(v=self.raw, atom_array=aa)
        return tokens


def from_any_(v: Any, atom_array: AtomArray):
    data_norm = canonicalize_(v, atom_array)

    # Canonicalize dictionaries to SelectionDict (I.e. convert "ALL" / "TIP" -> concrete atom names)
    data_split = {}
    mask = np.array([False] * len(atom_array))
    tokens = {}
    for idx, atm_names in data_norm.items():
        # Find atom array subset
        comp_mask = fetch_mask_from_idx(idx, atom_array=atom_array)
        token = atom_array[comp_mask]

        comp_mask_subset = get_name_mask(
            token.atom_name, atm_names, token.res_name[0]
        )  # [N_atoms_in_token,]

        # Split to atom names
        data_split[idx] = token.atom_name[comp_mask_subset].tolist()

        # Update mask & token dictionary
        mask[comp_mask] = comp_mask_subset
        tokens[idx] = token[comp_mask_subset]

    return (data_split, mask, tokens)


def canonicalize_(v, atom_array: AtomArray):
    # Canonicalize inputs to dictionaries of strings:
    # "A11-12" -> {"A11": "N,CA,C,...", "A12": "N,CA,C,..."}
    # True -> {"A1": "ALL", "A2": "ALL", ...}
    # False -> {"A1": "", "A2": "", ...}
    # "LIG" -> {"B1": "ALL", "C1": "ALL"}  (for two ligands named LIG)
    data = {}
    if isinstance(v, str):
        for component in unravel_components(
            v, atom_array=atom_array, allow_multiple_matches=True
        ):
            if (
                isinstance(component, str) and component[0].isalpha()
            ):  # filter on valid chain IDs
                data[component] = "ALL"

    elif isinstance(v, bool):
        starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            token = atom_array[start:stop]
            # All atoms selected for every token or None
            data[f"{token.chain_id[0]}{token.res_id[0]}"] = "ALL" if v else ""

    elif isinstance(v, dict):
        # Ensure all values of dictionaries are strings
        data = {}
        for k, vv in v.items():
            for component in unravel_components(
                k, atom_array=atom_array, allow_multiple_matches=True
            ):
                if isinstance(vv, list):
                    data[component] = ",".join(vv)
                else:
                    data[component] = vv
    else:
        raise ValueError(f"Cannot convert {type(v)} to InputSelection")

    return data
