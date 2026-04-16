import numpy as np
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.utils.token import apply_and_spread_token_wise


class CalculateRASA(Transform):
    """Transform for calculating relative SASA (RASA) for each atom in an AtomArray."""

    def __init__(
        self,
        probe_radius: float = 1.4,
        atom_radii: str | np.ndarray = "ProtOr",
        point_number: int = 100,
        requires_ligand=False,
    ):
        """
        probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
        atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr".
            "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
        point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.
        """
        self.probe_radius = probe_radius
        self.atom_radii = atom_radii
        self.point_number = point_number
        self.requires_ligand = requires_ligand

    def forward(self, data):
        atom_array = data["atom_array"]

        if not np.any(atom_array.is_ligand) and self.requires_ligand:
            return data

        # Calculate exact rasa
        rasa = calculate_atomwise_rasa(
            atom_array, self.probe_radius, self.atom_radii, self.point_number
        )
        atom_array.set_annotation("rasa", rasa)

        data["atom_array"] = atom_array
        return data


def discretize_rasa(atom_array, low=0, high=0.2, n_bins=3, keep_protein_motif=False):
    inclusion_mask = ~np.isnan(atom_array.rasa)
    inclusion_mask = inclusion_mask & atom_array.is_motif_token
    if not keep_protein_motif:
        inclusion_mask = inclusion_mask & ~atom_array.is_protein

    bin_edges = np.linspace(low, high, n_bins)  # e.g., [0.0, 0.1, 0.2]
    bins = (
        np.digitize(atom_array.rasa, bin_edges, right=False)
        - 1  # Subtract 1 since first bin would mean negative rasa!
    )  # bins in [0, n_bins-1]
    bins[~inclusion_mask] = n_bins  # Assign excluded atoms to an additional, unused bin
    return bins


class SetZeroOccOnDeltaRASA(Transform):
    """
    Recomputes RASA and sets zero-occupancy for those that have become significantly exposed

    Used to measure if the atomwise RASA changed during cropping
    """

    requires_previous_transforms = [CalculateRASA]
    incompatible_previous_transforms = [
        "PadWithVirtualAtoms",  # must have the same atom names
        "CreateDesignReferenceFeatures",
        "AggregateFeaturesLikeAF3WithoutMSA",
    ]

    def __init__(
        self,
        probe_radius: float = 1.4,
        atom_radii: str | np.ndarray = "ProtOr",
        point_number: int = 100,
    ):
        self.probe_radius = probe_radius
        self.atom_radii = atom_radii
        self.point_number = point_number

    def check_input(self, data):
        assert "rasa" in data["atom_array"].get_annotation_categories()

    def forward(self, data):
        atom_array = data["atom_array"]
        rasa_old = atom_array.rasa

        rasa_new = calculate_atomwise_rasa(
            atom_array, self.probe_radius, self.atom_radii, self.point_number
        )

        delta_rasa = np.clip(rasa_new, a_min=0, a_max=0.2) - np.clip(
            rasa_old, a_min=0, a_max=0.2
        )
        has_become_exposed = np.nan_to_num(delta_rasa) > 0.075
        token_has_become_exposed = apply_and_spread_token_wise(
            atom_array,
            has_become_exposed,
            function=lambda x: np.any(x),
        )
        is_sidechain = (
            ~np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])
            & atom_array.is_residue
        )

        # Set zero occupancy for sidechains only
        atom_has_become_exposed = token_has_become_exposed & is_sidechain

        atom_array.occupancy[atom_has_become_exposed] = 0.0
        # atom_array.res_name[token_has_become_exposed] = "UNK"

        data["atom_array"] = atom_array

        return data
