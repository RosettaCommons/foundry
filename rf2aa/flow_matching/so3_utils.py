import logging
import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def scale_rotmat(
    rotation_matrix: torch.Tensor, scalar: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:
    """
    Scale rotation matrix. This is done by converting it to vector representation,
    scaling the length of the vector and converting back to matrix representation.

    Args:
        rotation_matrix: Rotation matrices.
        scalar: Scalar values used for scaling. Should have one fewer dimension than the
            rotation matrices for correct broadcasting.
        tol: Numerical offset for stability.

    Returns:
        Scaled rotation matrix.
    """
    # Check whether dimensions match.
    assert rotation_matrix.ndim - 1 == scalar.ndim
    scaled_rmat = rotvec_to_rotmat(rotmat_to_rotvec(rotation_matrix) * scalar, tol=tol)
    return scaled_rmat


def _broadcast_identity(target: torch.Tensor) -> torch.Tensor:
    """
    Generate a 3 by 3 identity matrix and broadcast it to a batch of target matrices.

    Args:
        target (torch.Tensor): Batch of target 3 by 3 matrices.

    Returns:
        torch.Tensor: 3 by 3 identity matrices in the shapes of the target.
    """
    id3 = torch.eye(3, device=target.device, dtype=target.dtype)
    id3 = torch.broadcast_to(id3, target.shape)
    return id3


def skew_matrix_exponential_map_axis_angle(
    angles: torch.Tensor, skew_matrices: torch.Tensor
) -> torch.Tensor:
    """
    Compute the matrix exponential of a rotation in axis-angle representation with the axis in skew
    matrix representation form. Maps the rotation from the lie group to the rotation matrix
    representation. Uses Rodrigues' formula instead of `torch.linalg.matrix_exp` for better
    computational performance:

    .. math::

        \exp(\theta \mathbf{K}) = \mathbf{I} + \sin(\theta) \mathbf{K} + [1 - \cos(\theta)] \mathbf{K}^2

    Args:
        angles (torch.Tensor): Batch of rotation angles.
        skew_matrices (torch.Tensor): Batch of rotation axes in skew matrix (lie so(3)) basis.

    Returns:
        torch.Tensor: Batch of corresponding rotation matrices.
    """
    # Set up identity matrix and broadcast.
    id3 = _broadcast_identity(skew_matrices)

    # Broadcast angle vector to right dimensions
    angles = angles[..., None, None]

    exp_skew = (
        id3
        + torch.sin(angles) * skew_matrices
        + (1.0 - torch.cos(angles))
        * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def skew_matrix_exponential_map(
    angles: torch.Tensor, skew_matrices: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the matrix exponential of a rotation vector in skew matrix representation. Maps the
    rotation from the lie group to the rotation matrix representation. Uses the following form of
    Rodrigues' formula instead of `torch.linalg.matrix_exp` for better computational performance
    (in this case the skew matrix already contains the angle factor):

    .. math ::

        \exp(\mathbf{K}) = \mathbf{I} + \frac{\sin(\theta)}{\theta} \mathbf{K} + \frac{1-\cos(\theta)}{\theta^2} \mathbf{K}^2

    This form has the advantage, that Taylor expansions can be used for small angles (instead of
    having to compute the unit length axis by dividing the rotation vector by small angles):

    .. math ::

        \frac{\sin(\theta)}{\theta} \approx 1 - \frac{\theta^2}{6}
        \frac{1-\cos(\theta)}{\theta^2} \approx \frac{1}{2} - \frac{\theta^2}{24}

    Args:
        angles (torch.Tensor): Batch of rotation angles.
        skew_matrices (torch.Tensor): Batch of rotation axes in skew matrix (lie so(3)) basis.

    Returns:
        torch.Tensor: Batch of corresponding rotation matrices.
    """
    # Set up identity matrix and broadcast.
    id3 = _broadcast_identity(skew_matrices)

    # Broadcast angles and pre-compute square.
    angles = angles[..., None, None]
    angles_sq = angles.square()

    # Get standard terms.
    sin_coeff = torch.sin(angles) / angles
    cos_coeff = (1.0 - torch.cos(angles)) / angles_sq
    # Use second order Taylor expansion for values close to zero.
    sin_coeff_small = 1.0 - angles_sq / 6.0
    cos_coeff_small = 0.5 - angles_sq / 24.0

    mask_zero = torch.abs(angles) < tol
    sin_coeff = torch.where(mask_zero, sin_coeff_small, sin_coeff)
    cos_coeff = torch.where(mask_zero, cos_coeff_small, cos_coeff)

    # Compute matrix exponential using Rodrigues' formula.
    exp_skew = (
        id3
        + sin_coeff * skew_matrices
        + cos_coeff * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def rotvec_to_rotmat(rotation_vectors: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    """
    Convert rotation vectors to rotation matrix representation. The length of the rotation vector
    is the angle of rotation, the unit vector the rotation axis.

    Args:
        rotation_vectors (torch.Tensor): Batch of rotation vectors.
        tol: small offset for numerical stability.

    Returns:
        torch.Tensor: Rotation in rotation matrix representation.
    """
    # Compute rotation angle as vector norm.
    rotation_angles = torch.norm(rotation_vectors, dim=-1)

    # Map axis to skew matrix basis.
    skew_matrices = vector_to_skew_matrix(rotation_vectors)

    # Compute rotation matrices via matrix exponential.
    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices, tol=tol)

    return rotation_matrices


def rotmat_to_rotvec(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to rotation vectors (logarithmic map from SO(3) to so(3)).
    The standard logarithmic map can be derived from Rodrigues' formula via Taylor approximation
    (in this case operating on the vector coefficients of the skew so(3) basis).

    ..math ::

        \left[\log(\mathbf{R})\right]^\lor = \frac{\theta}{2\sin(\theta)} \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    This formula has problems at 1) angles theta close or equal to zero and 2) at angles close and
    equal to pi.

    To improve numerical stability for case 1), the angle term at small or zero angles is
    approximated by its truncated Taylor expansion:

    .. math ::

        \left[\log(\mathbf{R})\right]^\lor \approx \frac{1}{2} (1 + \frac{\theta^2}{6}) \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    For angles close or equal to pi (case 2), the outer product relation can be used to obtain the
    squared rotation vector:

    .. math :: \omega \otimes \omega = \frac{1}{2}(\mathbf{I} + R)

    Taking the root of the diagonal elements recovers the normalized rotation vector up to the signs
    of the component. The latter can be obtained from the off-diagonal elements.

    Adapted from https://github.com/jasonkyuyim/se3_diffusion/blob/2cba9e09fdc58112126a0441493b42022c62bbea/data/so3_utils.py
    which was adapted from https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py
    with heavy help from https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

    Args:
        rotation_matrices (torch.Tensor): Input batch of rotation matrices.

    Returns:
        torch.Tensor: Batch of rotation vectors.
    """
    # Get angles and sin/cos from rotation matrix.
    angles, angles_sin, _ = angle_from_rotmat(rotation_matrices)
    # Compute skew matrix representation and extract so(3) vector components.
    vector = skew_matrix_to_vector(rotation_matrices - rotation_matrices.transpose(-2, -1))

    # Three main cases for angle theta, which are captured
    # 1) Angle is 0 or close to zero -> use Taylor series for small values / return 0 vector.
    mask_zero = torch.isclose(angles, torch.zeros_like(angles)).to(angles.dtype)
    # 2) Angle is close to pi -> use outer product relation.
    mask_pi = torch.isclose(angles, torch.full_like(angles, np.pi), atol=1e-2).to(angles.dtype)
    # 3) Angle is unproblematic -> use the standard formula.
    mask_else = (1 - mask_zero) * (1 - mask_pi)

    # Compute case dependent pre-factor (1/2 for angle close to 0, angle otherwise).
    numerator = mask_zero / 2.0 + angles * mask_else
    # The Taylor expansion used here is actually the inverse of the Taylor expansion of the inverted
    # fraction sin(x) / x which gives better accuracy over a wider range (hence the minus and
    # position in denominator).
    denominator = (
        (1.0 - angles**2 / 6.0) * mask_zero  # Taylor expansion for small angles.
        + 2.0 * angles_sin * mask_else  # Standard formula.
        + mask_pi  # Avoid zero division at angle == pi.
    )
    prefactor = numerator / denominator
    vector = vector * prefactor[..., None]

    # For angles close to pi, derive vectors from their outer product (ww' = 1 + R).
    id3 = _broadcast_identity(rotation_matrices)
    skew_outer = (id3 + rotation_matrices) / 2.0
    # Ensure diagonal is >= 0 for square root (uses identity for masking).
    skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3

    # Get basic rotation vector as sqrt of diagonal (is unit vector).
    vector_pi = torch.sqrt(torch.diagonal(torch.clamp(skew_outer, min=1e-8), dim1=-2, dim2=-1))

    # Compute the signs of vector elements (up to a global phase).
    # Fist select indices for outer product slices with the largest norm.
    signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
    # Select rows of outer product and determine signs.
    signs_line = torch.take_along_dim(skew_outer, dim=-2, indices=signs_line_idx[..., None, None])
    signs_line = signs_line.squeeze(-2)
    signs = torch.sign(signs_line)

    # Apply signs and rotation vector.
    vector_pi = vector_pi * angles[..., None] * signs

    # Fill entries for angle == pi in rotation vector (basic vector has zero entries at this point).
    vector = vector + vector_pi * mask_pi[..., None]

    return vector


def angle_from_rotmat(
    rotation_matrices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rotation angles (as well as their sines and cosines) encoded by rotation matrices.
    Uses atan2 for better numerical stability for small angles.

    Args:
        rotation_matrices (torch.Tensor): Batch of rotation matrices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batch of computed angles, sines of the
          angles and cosines of angles.
    """
    # Compute sine of angles (uses the relation that the unnormalized skew vector generated by a
    # rotation matrix has the length 2*sin(theta))
    skew_matrices = rotation_matrices - rotation_matrices.transpose(-2, -1)
    skew_vectors = skew_matrix_to_vector(skew_matrices)
    angles_sin = torch.norm(skew_vectors, dim=-1) / 2.0
    # Compute the cosine of the angle using the relation cos theta = 1/2 * (Tr[R] - 1)
    angles_cos = (torch.einsum("...ii", rotation_matrices) - 1.0) / 2.0

    # Compute angles using the more stable atan2
    angles = torch.atan2(angles_sin, angles_cos)

    return angles, angles_sin, angles_cos


def vector_to_skew_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """
    Map a vector into the corresponding skew matrix so(3) basis.
    ```
                [  0 -z  y]
    [x,y,z] ->  [  z  0 -x]
                [ -y  x  0]
    ```

    Args:
        vectors (torch.Tensor): Batch of vectors to be mapped to skew matrices.

    Returns:
        torch.Tensor: Vectors in skew matrix representation.
    """
    # Generate empty skew matrices.
    skew_matrices = torch.zeros((*vectors.shape, 3), device=vectors.device, dtype=vectors.dtype)

    # Populate positive values.
    skew_matrices[..., 2, 1] = vectors[..., 0]
    skew_matrices[..., 0, 2] = vectors[..., 1]
    skew_matrices[..., 1, 0] = vectors[..., 2]

    # Generate skew symmetry.
    skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)

    return skew_matrices


def skew_matrix_to_vector(skew_matrices: torch.Tensor) -> torch.Tensor:
    """
    Extract a rotation vector from the so(3) skew matrix basis.

    Args:
        skew_matrices (torch.Tensor): Skew matrices.

    Returns:
        torch.Tensor: Rotation vectors corresponding to skew matrices.
    """
    vectors = torch.zeros_like(skew_matrices[..., 0])
    vectors[..., 0] = skew_matrices[..., 2, 1]
    vectors[..., 1] = skew_matrices[..., 0, 2]
    vectors[..., 2] = skew_matrices[..., 1, 0]
    return vectors


def _rotquat_to_axis_angle(
    rotation_quaternions: torch.Tensor, tol: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Auxiliary routine for computing rotation angle and rotation axis from unit quaternions. To avoid
    complications, rotations vectors with angles below `tol` are set to zero.

    Args:
        rotation_quaternions (torch.Tensor): Rotation quaternions in [r, i, j, k] format.
        tol (float, optional): Threshold for small rotations. Defaults to 1e-7.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotation angles and axes.
    """
    # Compute rotation axis and normalize (accounting for small length axes).
    rotation_axes = rotation_quaternions[..., 1:]
    rotation_axes_norms = torch.norm(rotation_axes, dim=-1)

    # Compute rotation angle via atan2
    rotation_angles = 2.0 * torch.atan2(rotation_axes_norms, rotation_quaternions[..., 0])

    # Save division.
    rotation_axes = rotation_axes / (rotation_axes_norms[:, None] + tol)
    return rotation_angles, rotation_axes


def rotquat_to_rotvec(rotation_quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation vectors.

    Args:
        rotation_quaternions (torch.Tensor): Input quaternions in [r,i,j,k] format.

    Returns:
        torch.Tensor: Rotation vectors.
    """
    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    rotation_vectors = rotation_axes * rotation_angles[..., None]
    return rotation_vectors


def rotquat_to_rotmat(rotation_quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion to rotation matrix.

    Args:
        rotation_quaternions (torch.Tensor): Input quaternions in [r,i,j,k] format.

    Returns:
        torch.Tensor: Rotation matrices.
    """
    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    skew_matrices = vector_to_skew_matrix(rotation_axes * rotation_angles[..., None])
    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices)
    return rotation_matrices


def apply_rotvec_to_rotmat(
    rotation_matrices: torch.Tensor,
    rotation_vectors: torch.Tensor,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Update a rotation encoded in a rotation matrix with a rotation vector.

    Args:
        rotation_matrices: Input batch of rotation matrices.
        rotation_vectors: Input batch of rotation vectors.
        tol: Small offset for numerical stability.

    Returns:
        Updated rotation matrices.
    """
    # Convert vector to matrices.
    rmat_right = rotvec_to_rotmat(rotation_vectors, tol=tol)
    # Accumulate rotation.
    rmat_rotated = torch.einsum("...ij,...jk->...ik", rotation_matrices, rmat_right)
    return rmat_rotated


def rotmat_to_skew_matrix(mat: torch.Tensor) -> torch.Tensor:
    """
    Generates skew matrix for corresponding rotation matrix.

    Args:
        mat (torch.Tensor): Batch of rotation matrices.

    Returns:
        torch.Tensor: Skew matrices in the shapes of mat.
    """
    vec = rotmat_to_rotvec(mat)
    return vector_to_skew_matrix(vec)


def skew_matrix_to_rotmat(skew: torch.Tensor) -> torch.Tensor:
    """
    Generates rotation matrix for corresponding skew matrix.

    Args:
        skew (torch.Tensor): Batch of target 3 by 3 skew symmetric matrices.

    Returns:
        torch.Tensor: Rotation matrices in the shapes of skew.
    """
    vec = skew_matrix_to_vector(skew)
    return rotvec_to_rotmat(vec)


def local_log(point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    """
    Matrix logarithm. Computes left-invariant vector field of beinging base_point to point
    on the manifold. Follows the signature of geomstats' equivalent function.
    https://geomstats.github.io/api/geometry.html#geomstats.geometry.lie_group.MatrixLieGroup.log

    Args:
        point (torch.Tensor): Batch of rotation matrices to compute vector field at.
        base_point (torch.Tensor): Transport coordinates to take matrix logarithm.

    Returns:
        torch.Tensor: Skew matrix that holds the vector field (in the tangent space).
    """
    return rotmat_to_skew_matrix(rot_mult(rot_transpose(base_point), point))


def multidim_trace(mat: torch.Tensor) -> torch.Tensor:
    """Take the trace of a matrix with leading dimensions."""
    return torch.einsum("...ii->...", mat)


def geodesic_dist(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the geodesic distance of two rotation matrices.

    Args:
        mat_1 (torch.Tensor): First rotation matrix.
        mat_2 (torch.Tensor): Second rotation matrix.

    Returns:
        Scalar for the geodesic distance between mat_1 and mat_2 with the same
        leading (i.e. batch) dimensions.
    """
    A = rotmat_to_skew_matrix(rot_mult(rot_transpose(mat_1), mat_2))
    return torch.sqrt(multidim_trace(rot_mult(A, rot_transpose(A))))


def rot_transpose(mat: torch.Tensor) -> torch.Tensor:
    """Take the transpose of the last two dimensions."""
    return torch.transpose(mat, -1, -2)


def rot_mult(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
    """Matrix multiply two rotation matrices with leading dimensions."""
    return torch.einsum("...ij,...jk->...ik", mat_1, mat_2)


def calc_rot_vf(mat_t: torch.Tensor, mat_1: torch.Tensor) -> torch.Tensor:
    """
    Computes the vector field Log_{mat_t}(mat_1).

    Args:
        mat_t (torch.Tensor): base point to compute vector field at.
        mat_1 (torch.Tensor): target rotation.

    Returns:
        Rotation vector representing the vector field.
    """
    return rotmat_to_rotvec(rot_mult(rot_transpose(mat_t), mat_1))


def geodesic_t(t: float, mat: torch.Tensor, base_mat: torch.Tensor, rot_vf=None) -> torch.Tensor:
    """
    Computes the geodesic at time t. Specifically, R_t = Exp_{base_mat}(t * Log_{base_mat}(mat)).

    Args:
        t: time along geodesic.
        mat: target points on manifold.
        base_mat: source point on manifold.

    Returns:
        Point along geodesic starting at base_mat and ending at mat.
    """
    if rot_vf is None:
        rot_vf = calc_rot_vf(base_mat, mat)
    mat_t = rotvec_to_rotmat(t * rot_vf)
    if base_mat.shape != mat_t.shape:
        raise ValueError(
            f'Incompatible shapes: base_mat={base_mat.shape}, mat_t={mat_t.shape}')
    return torch.einsum("...ij,...jk->...ik", base_mat, mat_t)


class SO3LookupCache:
    def __init__(
        self,
        cache_dir: str,
        cache_file: str,
        overwrite: bool = False,
    ) -> None:
        """
        Auxiliary class for handling storage / loading of SO(3) lookup tables in npz format.

        Args:
            cache_dir: Path to the cache directory.
            cache_file: Basic file name of the cache file.
            overwrite: Whether existing cache files should be overwritten if requested.
        """
        if not cache_file.endswith(".npz"):
            raise ValueError("Filename should have '.npz' extension.")
        self.cache_file = cache_file
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_file)
        self.overwrite = overwrite

    @property
    def path_exists(self) -> bool:
        return os.path.exists(self.cache_path)

    @property
    def dir_exists(self) -> bool:
        return os.path.exists(self.cache_dir)

    def delete_cache(self) -> None:
        """
        Delete the cache file.
        """
        if self.path_exists:
            os.remove(self.cache_path)

    def load_cache(self) -> Dict[str, torch.Tensor]:
        """
        Load data from the cache file.

        Returns:
            Dictionary of loaded data tensors.
        """
        if self.path_exists:
            # Load data and convert to torch tensors.
            npz_data = np.load(self.cache_path)
            torch_dict = {f: torch.from_numpy(npz_data[f]) for f in npz_data.files}
            logger.info(f"Data loaded from {self.cache_path}")
            return torch_dict
        else:
            raise ValueError(f"No cache data found at {self.cache_path}.")

    def save_cache(self, data: Dict[str, torch.Tensor]) -> None:
        """
        Save a dictionary of tensors to the cache file. If overwrite is set to True, an existing
        file is overwritten, otherwise a warning is raised and the file is not modified.

        Args:
            data: Dictionary of tensors that should be saved to the cache.
        """
        if not self.dir_exists:
            os.makedirs(self.cache_dir)

        if self.path_exists:
            if self.overwrite:
                logger.info("Overwriting cache ...")
                self.delete_cache()
            else:
                logger.warn(
                    f"Cache at {self.cache_path} exits and overwriting disabled. Doing nothing."
                )
        else:
            # Move everything to CPU and numpy and store.
            logger.info(f"Data saved to {self.cache_path}")
            numpy_dict = {k: v.detach().cpu().numpy() for k, v in data.items()}
            np.savez(self.cache_path, **numpy_dict)


class BaseSampleSO3(nn.Module):
    so3_type: str = "base"  # cache basename

    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        tol: float = 1e-7,
        interpolate: bool = True,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
        device: str = 'cpu',
    ) -> None:
        """
        Base torch.nn module for sampling rotations from the IGSO(3) distribution. Samples are
        created by uniformly sampling a rotation axis and using inverse transform sampling for
        the angles. The latter uses the associated SO(3) cumulative probability distribution
        function (CDF) and a uniform distribution [0,1] as described in [#leach2022_1]_. CDF values
        are obtained by numerically integrating the probability distribution evaluated on a grid of
        angles and noise levels and stored in a lookup table. Linear interpolation is used to
        approximate continuos sampling of the function. Angles are discretized in an interval [0,pi]
        and the grid can be squashed to have higher resolutions at low angles by taking different
        powers. Since sampling relies on tabulated values of the CDF and indexing in the form of
        `torch.bucketize`, gradients are not supported.

        Args:
            num_omega (int): Number of discrete angles used for generating the lookup table.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.
            omega_exponent (int, optional): Make the angle grid denser for smaller angles by taking
              its power with the provided number. Defaults to 3.
            tol (float, optional): Small value for numerical stability. Defaults to 1e-7.
            interpolate (bool, optional): If enables, perform linear interpolation of the angle CDF
              to sample angles. Otherwise the closest tabulated point is returned. Defaults to True.
            cache_dir: Path to an optional cache directory. If set to None, lookup tables are
              computed on the fly.
            overwrite_cache: If set to true, existing cache files are overwritten. Can be used for
              updating stale caches.

        References
        ----------
        .. [#leach2022_1] Leach, Schmon, Degiacomi, Willcocks:
           Denoising diffusion probabilistic models on so (3) for rotational alignment.
           ICLR 2022 Workshop on Geometrical and Topological Representation Learning. 2022.
        """
        super().__init__()
        self.num_omega = num_omega
        self.omega_exponent = omega_exponent
        self.tol = tol
        self.interpolate = interpolate
        self.device = device
        self.register_buffer("sigma_grid", sigma_grid, persistent=False)

        # Generate / load lookups and store in non-persistent buffers.
        omega_grid, cdf_igso3 = self._setup_lookup(sigma_grid, cache_dir, overwrite_cache)
        self.register_buffer("omega_grid", omega_grid, persistent=False)
        self.register_buffer("cdf_igso3", cdf_igso3, persistent=False)

    def _setup_lookup(
        self,
        sigma_grid: torch.Tensor,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Master function for setting up the lookup tables. These can either be loaded from a npz
        cache file or computed on the fly. Lookup tables will always be created and stored in double
        precision. Casting to the target dtype is done at the end of the function.

        Args:
            sigma_grid: Grid of sigma values used for computing the lookup tables.
            cache_dir: Path to the cache directory.
            overwrite_cache: If set to true, an existing cache is overwritten. Can be used for
              updating stale caches.

        Returns:
            Grid of angle values and SO(3) cumulative distribution function.
        """
        if cache_dir is not None:
            cache_name = self._get_cache_name()
            cache = SO3LookupCache(cache_dir, cache_name, overwrite=True)

            # If cache dir is provided, check whether the necessary cache exists and whether it
            # should be overwritten.
            if cache.path_exists and not overwrite_cache:
                # Load data from cache.
                cache_data = cache.load_cache()
                omega_grid = cache_data["omega_grid"]
                cdf_igso3 = cache_data["cdf_igso3"]
            else:
                # Store data in cache (overwrite if requested).
                omega_grid, cdf_igso3 = self._generate_lookup(sigma_grid)
                cache.save_cache({"omega_grid": omega_grid, "cdf_igso3": cdf_igso3})
        else:
            # Other wise just generate the tables.
            omega_grid, cdf_igso3 = self._generate_lookup(sigma_grid)

        return omega_grid.to(sigma_grid.dtype), cdf_igso3.to(sigma_grid.dtype)

    def _get_cache_name(self) -> str:
        """
        Auxiliary function for determining the cache file name based on the parameters (sigma,
        omega, l, etc.) used for generating the lookup tables.

        Returns:
            Base name of the cache file.
        """
        cache_name = "cache_{:s}_s{:04.3f}-{:04.3f}-{:d}_o{:d}-{:d}.npz".format(
            self.so3_type,
            torch.min(self.sigma_grid).cpu().item(),
            torch.max(self.sigma_grid).cpu().item(),
            self.sigma_grid.shape[0],
            self.num_omega,
            self.omega_exponent,
        )
        return cache_name

    def get_sigma_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous sigmas to the indices of the closest tabulated values.

        Args:
            sigma (torch.Tensor): IGSO3 std devs.

        Returns:
            torch.Tensor: Index tensor mapping the provided sigma values to the internal lookup
              table.
        """
        return torch.bucketize(sigma, self.sigma_grid)

    def expansion_function(
        self, omega_grid: torch.Tensor, sigma_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Function for generating the angle probability distribution. Should return a 2D tensor with
        values for the std dev at the first dimension (rows) and angles at the second
        (columns).

        Args:
            omega_grid (torch.Tensor): Grid of angle values.
            sigma_grid (torch.Tensor): IGSO3 std devs.

        Returns:
            torch.Tensor: Distribution for angles discretized on a 2D grid.
        """
        raise NotImplementedError

    @torch.no_grad()
    def _generate_lookup(self, sigma_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the lookup table for sampling from the target SO(3) CDF. The table is 2D, with the
        rows corresponding to different sigma values and the columns with angles computed on a grid.
        Variance is scaled by a factor of 1/2 to account for the deacceleration of time in the
        diffusion process due to the choice of SO(3) basis and guarantee time-reversibility (see
        appendix E.3 in [#yim2023_2]_). The returned tables are double precision and will be cast
        to the target dtype in `_setup_lookup`.

        Args:
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the grid used to compute the angles
              and the associated lookup table.

        References
        ----------
        .. [#yim2023_2] Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola:
           SE(3) diffusion model with application to protein backbone generation.
           arXiv preprint arXiv:2302.02277. 2023.
        """

        current_device = sigma_grid.device
        sigma_grid_tmp = sigma_grid.to(torch.float64)

        # If cuda is available, initialize everything on GPU.
        # Even if Pytorch Lightning usually handles GPU allocation after initialization, this is
        # required to initialize the module in GPU reducing the initializaiton time by orders of magnitude.
        if torch.cuda.is_available():
            sigma_grid_tmp = sigma_grid_tmp.to(device=self.device)

        # Set up grid for angle resolution. Convert to double precision for better handling of numerics.
        omega_grid = torch.linspace(0.0, 1, self.num_omega + 1).to(sigma_grid_tmp)

        # If requested, increase sample density for lower values
        omega_grid = omega_grid**self.omega_exponent

        omega_grid = omega_grid * np.pi

        # Compute the expansion for all omegas and sigmas.
        pdf_igso3 = self.expansion_function(omega_grid, sigma_grid_tmp)

        # Apply the pre-factor from USO(3).
        pdf_igso3 = pdf_igso3 * (1.0 - torch.cos(omega_grid)) / np.pi

        # Compute the cumulative probability distribution.
        cdf_igso3 = integrate_trapezoid_cumulative(pdf_igso3, omega_grid)
        # Normalize integral area to 1.
        cdf_igso3 = cdf_igso3 / cdf_igso3[:, -1][:, None]

        # Move back to original device.
        cdf_igso3 = cdf_igso3.to(device=current_device)
        omega_grid = omega_grid.to(device=current_device)

        return omega_grid[1:].to(sigma_grid.dtype), cdf_igso3.to(sigma_grid.dtype)

    def sample(self, sigma: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the target SO(3) distribution by sampling a rotation axis angle,
        which are then combined into a rotation vector and transformed into the corresponding
        rotation matrix via an exponential map.

        Args:
            sigma_indices (torch.Tensor): Indices of the IGSO3 std devs for which to take samples.
            num_samples (int): Number of angle samples to take for each std dev

        Returns:
            torch.Tensor: Sampled rotations in matrix representation with dimensions
              [num_sigma x num_samples x 3 x 3].
        """

        vectors = self.sample_vector(sigma.shape[0], num_samples)
        angles = self.sample_angle(sigma, num_samples)

        # Do postprocessing on angles.
        angles = self._process_angles(sigma, angles)

        rotation_vectors = vectors * angles[..., None]

        rotation_matrices = rotvec_to_rotmat(rotation_vectors, tol=self.tol)
        return rotation_matrices

    def _process_angles(self, sigma: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary function for performing additional processing steps on the sampled angles. One
        example would be to ensure sampled angles are 0 for a std dev of 0 for IGSO(3).

        Args:
            sigma (torch.Tensor): Current values of sigma.
            angles (torch.Tensor): Sampled angles.

        Returns:
            torch.Tensor: Processed sampled angles.
        """
        return angles

    def sample_vector(self, num_sigma: int, num_samples: int) -> torch.Tensor:
        """
        Uniformly sample rotation axis for constructing the overall rotation.

        Args:
            num_sigma (int): Number of samples to draw for each std dev.
            num_samples (int): Number of angle samples to take for each std dev.

        Returns:
            torch.Tensor: Batch of rotation axes with dimensions [num_sigma x num_samples x 3].
        """
        vectors = torch.randn(num_sigma, num_samples, 3, device=self.sigma_grid.device)
        vectors = vectors / torch.norm(vectors, dim=2, keepdim=True)
        return vectors

    def sample_angle(self, sigma: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Create a series of samples from the IGSO(3) angle distribution.

        Args:
            sigma_indices (torch.Tensor): Indices of the IGSO3 std deves for which to
              take samples.
            num_samples (int): Number of angle samples to take for each std dev.

        Returns:
            torch.Tensor: Collected samples, will have the dimension [num_sigma x num_samples].
        """
        # Convert sigmas to respective indices for lookup table.
        sigma_indices = self.get_sigma_idx(sigma)
        # Get relevant sigma slices from stored CDFs.
        cdf_tmp = self.cdf_igso3[sigma_indices, :]

        # Draw from uniform distribution.
        p_uniform = torch.rand((*sigma_indices.shape, *[num_samples]), device=sigma_indices.device)

        # Determine indices for CDF.
        idx_stop = torch.sum(cdf_tmp[..., None] < p_uniform[:, None, :], dim=1).long()
        idx_start = torch.clamp(idx_stop - 1, min=0)

        if not self.interpolate:
            omega = torch.gather(cdf_tmp, dim=1, index=idx_stop)
        else:
            # Get CDF values.
            cdf_start = torch.gather(cdf_tmp, dim=1, index=idx_start)
            cdf_stop = torch.gather(cdf_tmp, dim=1, index=idx_stop)

            # Compute weights for linear interpolation.
            cdf_delta = torch.clamp(cdf_stop - cdf_start, min=self.tol)
            cdf_weight = torch.clamp((p_uniform - cdf_start) / cdf_delta, min=0.0, max=1.0)

            # Get angle range for interpolation.
            omega_start = self.omega_grid[idx_start]
            omega_stop = self.omega_grid[idx_stop]

            # Interpolate.
            omega = torch.lerp(omega_start, omega_stop, cdf_weight)

        return omega


class SampleIGSO3(BaseSampleSO3):
    so3_type = "igso3"  # cache basename

    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        tol: float = 1e-7,
        interpolate: bool = True,
        l_max: int = 1000,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
        device: str = 'cpu',
    ) -> None:
        """
        Module for sampling rotations from the IGSO(3) distribution using the explicit series
        expansion.  Samples are created using inverse transform sampling based on the associated
        cumulative probability distribution function (CDF) and a uniform distribution [0,1] as
        described in [#leach2022_2]_. CDF values are obtained by numerically integrating the
        probability distribution evaluated on a grid of angles and noise levels and stored in a
        lookup table.  Linear interpolation is used to approximate continuos sampling of the
        function. Angles are discretized in an interval [0,pi] and the grid can be squashed to have
        higher resolutions at low angles by taking different powers.
        Since sampling relies on tabulated values of the CDF and indexing in the form of
        `torch.bucketize`, gradients are not supported.

        Args:
            num_omega (int): Number of discrete angles used for generating the lookup table.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.
            omega_exponent (int, optional): Make the angle grid denser for smaller angles by taking
              its power with the provided number. Defaults to 3.
            tol (float, optional): Small value for numerical stability. Defaults to 1e-7.
            interpolate (bool, optional): If enables, perform linear interpolation of the angle CDF
              to sample angles. Otherwise the closest tabulated point is returned. Defaults to True.
            l_max (int, optional): Maximum number of terms used in the series expansion.
            cache_dir: Path to an optional cache directory. If set to None, lookup tables are
              computed on the fly.
            overwrite_cache: If set to true, existing cache files are overwritten. Can be used for
              updating stale caches.

        References
        ----------
        .. [#leach2022_2] Leach, Schmon, Degiacomi, Willcocks:
           Denoising diffusion probabilistic models on so (3) for rotational alignment.
           ICLR 2022 Workshop on Geometrical and Topological Representation Learning. 2022.
        """
        self.l_max = l_max
        super().__init__(
            num_omega=num_omega,
            sigma_grid=sigma_grid,
            omega_exponent=omega_exponent,
            tol=tol,
            interpolate=interpolate,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            device=device,
        )

    def _get_cache_name(self) -> str:
        """
        Auxiliary function for determining the cache file name based on the parameters (sigma,
        omega, l, etc.) used for generating the lookup tables.

        Returns:
            Base name of the cache file.
        """
        cache_name = "cache_{:s}_s{:04.3f}-{:04.3f}-{:d}_l{:d}_o{:d}-{:d}.npz".format(
            self.so3_type,
            torch.min(self.sigma_grid).cpu().item(),
            torch.max(self.sigma_grid).cpu().item(),
            self.sigma_grid.shape[0],
            self.l_max,
            self.num_omega,
            self.omega_exponent,
        )
        return cache_name

    def expansion_function(
        self,
        omega_grid: torch.Tensor,
        sigma_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use the truncated expansion of the IGSO(3) probability function to generate the lookup table.

        Args:
            omega_grid (torch.Tensor): Grid of angle values.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.

        Returns:
            torch.Tensor: IGSO(3) distribution for angles discretized on a 2D grid.
        """
        return generate_igso3_lookup_table(omega_grid, sigma_grid, l_max=self.l_max, tol=self.tol)

    def _process_angles(self, sigma: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Ensure sampled angles are 0 for small noise levels in IGSO(3). (Series expansion gives
        uniform probability distribution.)

        Args:
            sigma (torch.Tensor): Current values of sigma.
            angles (torch.Tensor): Sampled angles.

        Returns:
            torch.Tensor: Processed sampled angles.
        """
        angles = torch.where(
            sigma[..., None] < self.tol,
            torch.zeros_like(angles),
            angles,
        )
        return angles


class SampleUSO3(BaseSampleSO3):
    so3_type = "uso3"  # cache basename

    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        tol: float = 1e-7,
        interpolate: bool = True,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
    ) -> None:
        """
        Module for sampling rotations from the USO(3) distribution. Can be used to generate initial
        unbiased samples in the reverse process.  Samples are created using inverse transform
        sampling based on the associated cumulative probability distribution function (CDF) and a
        uniform distribution [0,1] as described in [#leach2022_4]_. CDF values are obtained by
        numerically integrating the probability distribution evaluated on a grid of angles and noise
        levels and stored in a lookup table.  Linear interpolation is used to approximate continuos
        sampling of the function. Angles are discretized in an interval [0,pi] and the grid can be
        squashed to have higher resolutions at low angles by taking different powers.
        Since sampling relies on tabulated values of the CDF and indexing in the form of
        `torch.bucketize`, gradients are not supported.

        Args:
            num_omega (int): Number of discrete angles used for generating the lookup table.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.
            omega_exponent (int, optional): Make the angle grid denser for smaller angles by taking
              its power with the provided number. Defaults to 3.
            tol (float, optional): Small value for numerical stability. Defaults to 1e-7.
            interpolate (bool, optional): If enables, perform linear interpolation of the angle CDF
              to sample angles. Otherwise the closest tabulated point is returned. Defaults to True.
            cache_dir: Path to an optional cache directory. If set to None, lookup tables are
              computed on the fly.
            overwrite_cache: If set to true, existing cache files are overwritten. Can be used for
              updating stale caches.

        References
        ----------
        .. [#leach2022_4] Leach, Schmon, Degiacomi, Willcocks:
           Denoising diffusion probabilistic models on so (3) for rotational alignment.
           ICLR 2022 Workshop on Geometrical and Topological Representation Learning. 2022.
        """
        super().__init__(
            num_omega=num_omega,
            sigma_grid=sigma_grid,
            omega_exponent=omega_exponent,
            tol=tol,
            interpolate=interpolate,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
        )

    def get_sigma_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(sigma).long()

    def sample_shape(self, num_sigma: int, num_samples: int) -> torch.Tensor:
        dummy_sigma = torch.zeros(num_sigma, device=self.sigma_grid.device)
        return self.sample(dummy_sigma, num_samples)

    def expansion_function(
        self,
        omega_grid: torch.Tensor,
        sigma_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        The probability density function of the uniform SO(3) distribution is the cosine scaling
        term (1-cos(omega))/pi which is applied automatically during sampling. This means, it is
        sufficient to return a tensor of ones to create the correct USO(3) lookup table.

        Args:
            omega_grid (torch.Tensor): Grid of angle values.
            sigma_grid (torch.Tensor): Grid of IGSO3 std devs.

        Returns:
            torch.Tensor: USO(3) distribution for angles discretized on a 2D grid.
        """
        return torch.ones(1, omega_grid.shape[0], device=omega_grid.device)


@torch.no_grad()
def integrate_trapezoid_cumulative(f_grid: torch.Tensor, x_grid: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary function for numerically integrating a discretized 1D function using the trapezoid
    rule. This is mainly used for computing the cumulative probability distributions for sampling
    from the IGSO(3) distribution. Works on a single 1D grid or a batch of grids.

    Args:
        f_grid (torch.Tensor): Discretized function values.
        x_grid (torch.Tensor): Discretized input values.

    Returns:
        torch.Tensor: Integrated function (not normalized).
    """
    f_sum = f_grid[..., :-1] + f_grid[..., 1:]
    delta_x = torch.diff(x_grid, dim=-1)
    integral = torch.cumsum((f_sum * delta_x[None, :]) / 2.0, dim=-1)
    return integral


class ScoreSO3(nn.Module):
    def __init__(
        self,
        num_omega: int,
        sigma_grid: torch.Tensor,
        omega_exponent: int = 3,
        l_max: int = 1000,
        tol: float = 1e-7,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
    ) -> None:
        """
        Module for computing scores for SO(3) diffusion (see e.g. [#yim2023_1]_). Score computation
        is performed in the rotation vector representation (tangent space). The score is computed
        as:

        .. math ::
            \nabla_\mathbf{q} = \frac{\mathbf{q}}{|\mathbf{q}|} \frac{\tfrac{\partial}{\partial |\mathbf{q}|} f_{\mathrm{IGSO(3)}}(|\mathbf{q}|)}{f_{\mathrm{IGSO(3)}}(|\mathbf{q}|)}

        The fraction of the IGSO(3) derivative and probability function is obtained via truncated
        series expansion.

        In addition to the scores, the module also pre-computes score scaling values for the loss
        function according to the equation:

        .. math ::
            \frac{1}{\lambda} = \sqrt{\frac{\sum_\omega \left[\partial_\omega \log f_\mathrm{IGSO(3)}(\omega)\right]^2 f_\mathrm{IGSO(3)}(\omega)}{3 \sum_\omega f_\mathrm{IGSO(3)}(\omega)}}

        These values are stored in a lookup table and can be retrieved based on the closest
        tabulated noise levels. While the score supports full gradients during optimization, the
        scaling is static.

        In general, it is recommended to use the same grid settings as for sampling the rotations.

        Args:
            num_omega: Number of discrete angles used for generating the lookup table.
            sigma_grid: Grid of IGSO3 std devs.
            omega_exponent: Make the angle grid denser for smaller angles by taking its power with
              the provided number.
            l_max: Maximum number of terms used in the series expansion.
            tol: Small value for numerical stability.
            cache_dir: Path to an optional cache directory. If set to None, lookup tables are
              computed on the fly.
            overwrite_cache: If set to true, existing cache files are overwritten. Can be used for
              updating stale caches.

        References
        ----------
        .. [#yim2023_1] Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola:
           SE(3) diffusion model with application to protein backbone generation.
           arXiv preprint arXiv:2302.02277. 2023.
        """
        super().__init__()

        # Store different parameters and grids.
        self.l_max = l_max
        self.tol = tol
        self.num_omega = num_omega
        self.omega_exponent = omega_exponent
        self.register_buffer("l_grid", torch.arange(l_max + 1))
        self.register_buffer("sigma_grid", sigma_grid)

        # Pre-compute score scaling and store in buffer.
        score_scaling = self._setup_scaling(sigma_grid, cache_dir, overwrite_cache)
        self.register_buffer("score_scaling", score_scaling, persistent=False)

    def _setup_scaling(
        self,
        sigma_grid: torch.Tensor,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False,
    ) -> torch.Tensor:
        """
        Master function for setting up the lookup table for SO(3) score scaling. It can either be
        loaded from a npz cache file or computed on the fly.

        Args:
            sigma_grid: Grid of sigma values used for computing the lookup tables.
            cache_dir: Path to the cache directory.
            overwrite_cache: If set to true, an existing cache is overwritten. Can be used for
              updating stale caches.

        Returns:
            SO(3) score scaling.
        """
        if cache_dir is not None:
            cache_name = self._get_cache_name()
            cache = SO3LookupCache(cache_dir, cache_name, overwrite=True)

            # If cache dir is provided, check whether the necessary cache exists and whether it
            # should be overwritten.
            if cache.path_exists and not overwrite_cache:
                # Load data from cache.
                cache_data = cache.load_cache()
                score_scaling = cache_data["score_scaling"]
            else:
                # Store data in cache (overwrite if requested).
                score_scaling = self._compute_score_scaling(sigma_grid)
                cache.save_cache({"score_scaling": score_scaling})
        else:
            # Other wise just generate the tables.
            score_scaling = self._compute_score_scaling(sigma_grid)
        return score_scaling.to(sigma_grid.dtype)

    def _get_cache_name(self) -> str:
        """
        Auxiliary function for determining the cache file name based on the parameters (sigma,
        omega, l, etc.) used for generating the lookup tables.

        Returns:
            Base name of the cache file.
        """
        cache_name = "cache_score-scaling_s{:04.3f}-{:04.3f}-{:d}_l{:d}_o{:d}-{:d}.npz".format(
            torch.min(self.sigma_grid).cpu().item(),
            torch.max(self.sigma_grid).cpu().item(),
            self.sigma_grid.shape[0],
            self.l_grid.shape[0],
            self.num_omega,
            self.omega_exponent,
        )
        return cache_name

    def get_sigma_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous sigmas to the indices of the closest tabulated values.

        Args:
            sigma: IGSO3 std devs

        Returns:
            Index tensor mapping the provided sigma values to the internal lookup table.
        """
        return torch.bucketize(sigma, self.sigma_grid)

    def get_score_scaling(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Get tabulated score scaling values for the closest tabulated noise levels. Since the score
        scaling is pre-computed and tabulated, no gradients will be propagated through it.

        Args:
            sigma: IGSO3 std devs.

        Returns:
            Tensor of score scaling values (same shape as sigma).
        """
        sigma_idx = self.get_sigma_idx(sigma)
        score_scaling = self.score_scaling[sigma_idx]
        return score_scaling.detach()

    def _compute_score_scaling(self, sigma_grid):
        """
        Compute loss scaling factors for the score based on the IGSO(3) probability density and the
        derivative of its logarithm. Variance is scaled by a factor of 1/2 to account for the
        deacceleration of time in the diffusion process due to the choice of SO(3) basis and
        guarantee time-reversibility (see appendix E.3 in [#yim2023_3]_).

        Args:
            sigma_grid: Grid of IGSO3 std devs.

        Returns:
            1D tensor with the same shape as `sigma_grid` containing the score scaling for
            different noise levels.

        References
        ----------
        .. [#yim2023_3] Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola:
           SE(3) diffusion model with application to protein backbone generation.
           arXiv preprint arXiv:2302.02277. 2023.
        """

        current_device = sigma_grid.device
        sigma_grid_tmp = sigma_grid.to(torch.float64)

        # If cuda is available, initialize everything on GPU.
        if torch.cuda.is_available():
            sigma_grid_tmp = sigma_grid_tmp.to(device="cuda")

        # Set up grid for angle resolution. Convert to double precision for better handling of
        # numerics.
        omega_grid = torch.linspace(0.0, 1, self.num_omega).to(sigma_grid_tmp)

        # Increase sample density for lower values
        omega_grid = omega_grid**self.omega_exponent
        omega_grid = omega_grid * np.pi

        # Compute the expansion for all omegas and sigmas.
        pdf_igso3 = generate_igso3_lookup_table(
            omega_grid, sigma_grid_tmp, l_max=self.l_max, tol=self.tol
        )
        # Apply the pre-factor from USO(3).
        prefactor_uniform = uniform_so3_density(omega_grid)
        pdf_igso3 = pdf_igso3 * prefactor_uniform[None, :]
        # Ensure everything is positive.
        pdf_igso3 = torch.abs(pdf_igso3)

        # Compute the derivative of the logarithm of the probability distribution.
        dlog_igso3 = generate_dlog_igso3_lookup_table(
            omega_grid, sigma_grid_tmp, l_max=self.l_max, tol=self.tol
        )

        # Compute scaling of the score for loss as expectation value.
        score_scaling = torch.sqrt(
            torch.sum(dlog_igso3**2 * pdf_igso3, dim=1)
            / (3.0 * torch.sum(pdf_igso3, dim=1) + self.tol)
        )

        score_scaling = score_scaling.to(device=current_device)

        return score_scaling.to(sigma_grid.dtype)

    def forward(self, sigma: torch.Tensor, rotation_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute the SO(3) score in rotation vector representation.

        Args:
            sigma: Tensor of std devs (with the shape [num_eps]).
            rotation_vectors: Rotation vectors with the shape [num_eps x 3].

        Returns:
            SO(3) score with the shape [num_eps x 3].
        """
        # Compute rotation angles as vector lengths.
        rotation_angles = torch.norm(rotation_vectors, dim=-1)
        # Compute the derivative term coming from the IGSO(3) expansion logarithm.
        dlog_igso3 = dlog_igso3_expansion(rotation_angles, sigma, self.l_grid, tol=self.tol)
        # Compute full score (small offset for numerical stability).
        score = rotation_vectors / (rotation_angles[..., None] + self.tol) * dlog_igso3[..., None]
        return score


def uniform_so3_density(omega: torch.Tensor) -> torch.Tensor:
    """
    Compute the density over the uniform angle distribution in SO(3).

    Args:
        omega: Angles in radians.

    Returns:
        Uniform distribution density.
    """
    return (1.0 - torch.cos(omega)) / np.pi


def igso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the IGSO(3) angle probability distribution function for pairs of angles and std dev
    levels. The expansion is computed using a grid of expansion orders ranging from 0 to l_max.

    This function approximates the power series in equation 5 of [#yim2023_3]_. With this
    parameterization, IGSO(3) agrees with the Brownian motion on SO(3) with t=sigma^2.

    Args:
        omega: Values of angles (1D tensor).
        sigma: Values of std dev of IGSO3 distribution (1D tensor of same shape as `omega`).
        l_grid: Tensor containing expansion orders (0 to l_max).
        tol: Small offset for numerical stability.

    Returns:
        IGSO(3) angle distribution function (without pre-factor for uniform SO(3) distribution).

    References
    ----------
    .. [#yim2023_3] Yim, Trippe, De Bortoli, Mathieu, Doucet, Barzilay, Jaakkola:
        SE(3) diffusion model with application to protein backbone generation.
        arXiv preprint arXiv:2302.02277. 2023.
    """
    # Pre-compute sine in denominator and clamp for stability.
    denom_sin = torch.sin(0.5 * omega)

    # Pre-compute terms that rely only on expansion orders.
    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = -l_grid * (l_grid + 1.0)

    # Pre-compute numerator of expansion which only depends on angles.
    numerator_sin = torch.sin((l_grid[None, :] + 1 / 2) * omega[:, None])

    # Pre-compute exponential term with (2l+1) prefactor.
    exponential_term = l_fac_1[None, :] * torch.exp(l_fac_2[None, :] * sigma[:, None] ** 2 / 2)

    # Compute series expansion
    f_igso = torch.sum(exponential_term * numerator_sin, dim=1)
    # For small omega, accumulate limit of sine fraction instead:
    # lim[x->0] sin((l+1/2)x) / sin(x/2) = 2l + 1
    f_limw = torch.sum(exponential_term * l_fac_1[None, :], dim=1)

    # Finalize expansion. Offset for stability can be added since omega is [0,pi] and sin(omega/2)
    # is positive in this interval.
    f_igso = f_igso / (denom_sin + tol)

    # Replace values at small omega with limit.
    f_igso = torch.where(omega <= tol, f_limw, f_igso)

    # Remove remaining numerical problems
    f_igso = torch.where(
        torch.logical_or(torch.isinf(f_igso), torch.isnan(f_igso)), torch.zeros_like(f_igso), f_igso
    )

    return f_igso


def digso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the derivative of the IGSO(3) angle probability distribution function with respect to
    the angles for pairs of angles and std dev levels. As in `igso3_expansion` a grid is used for the
    expansion levels. Evaluates the derivative directly in order to avoid second derivatives during
    backpropagation.

    The derivative of the angle-dependent part is computed as:

    .. math ::
        \frac{\partial}{\partial \omega} \frac{\sin((l+\tfrac{1}{2})\omega)}{\sin(\tfrac{1}{2}\omega)} = \frac{l\sin((l+1)\omega) - (l+1)\sin(l\omega)}{1 - \cos(\omega)}

    (obtained via quotient rule + different trigonometric identities).

    Args:
        omega: Values of angles (1D tensor).
        sigma: Values of IGSO3 distribution std devs (1D tensor of same shape as `omega`).
        l_grid: Tensor containing expansion orders (0 to l_max).
        tol: Small offset for numerical stability.

    Returns:
        IGSO(3) angle distribution derivative (without pre-factor for uniform SO(3) distribution).
    """
    denom_cos = 1.0 - torch.cos(omega)

    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = l_grid + 1.0
    l_fac_3 = -l_grid * l_fac_2

    # Pre-compute numerator of expansion which only depends on angles.
    numerator_sin = l_grid[None, :] * torch.sin(l_fac_2[None, :] * omega[:, None]) - l_fac_2[
        None, :
    ] * torch.sin(l_grid[None, :] * omega[:, None])

    # Compute series expansion
    df_igso = torch.sum(
        l_fac_1[None, :] * torch.exp(l_fac_3[None, :] * sigma[:, None] ** 2 / 2) * numerator_sin,
        dim=1,
    )

    # Finalize expansion. Offset for stability can be added since omega is [0,pi] and cosine term
    # is positive in this interval.
    df_igso = df_igso / (denom_cos + tol)

    # Replace values at small omega with limit (=0).
    df_igso = torch.where(omega <= tol, torch.zeros_like(df_igso), df_igso)

    # Remove remaining numerical problems
    df_igso = torch.where(
        torch.logical_or(torch.isinf(df_igso), torch.isnan(df_igso)),
        torch.zeros_like(df_igso),
        df_igso,
    )

    return df_igso


def dlog_igso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the derivative of the logarithm of the IGSO(3) angle distribution function for pairs of
    angles and std dev levels:

    .. math ::
        \frac{\partial}{\partial \omega} \log f(\omega) = \frac{\tfrac{\partial}{\partial \omega} f(\omega)}{f(\omega)}

    Required for SO(3) score computation.

    Args:
        omega: Values of angles (1D tensor).
        sigma: Values of IGSO3 std devs (1D tensor of same shape as `omega`).
        l_grid: Tensor containing expansion orders (0 to l_max).
        tol: Small offset for numerical stability.

    Returns:
        IGSO(3) angle distribution derivative (without pre-factor for uniform SO(3) distribution).
    """
    f_igso3 = igso3_expansion(omega, sigma, l_grid, tol=tol)
    df_igso3 = digso3_expansion(omega, sigma, l_grid, tol=tol)

    return df_igso3 / (f_igso3 + tol)


@torch.no_grad()
def generate_lookup_table(
    base_function: Callable,
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
):
    """
    Auxiliary function for generating a lookup table from IGSO(3) expansions and their derivatives.
    Takes a basic function and loops over different std dev levels.

    Args:
        base_function: Function used for setting up the lookup table.
        omega_grid: Grid of angle values ranging from [0,pi] (shape is[num_omega]).
        sigma_grid: Grid of IGSO3 std dev values (shape is [num_sigma]).
        l_max: Number of terms used in the series expansion.
        tol: Small value for numerical stability.

    Returns:
        Table of function values evaluated at different angles and std dev levels. The final shape is
        [num_sigma x num_omega].
    """
    # Generate grid of expansion orders.
    l_grid = torch.arange(l_max + 1, device=omega_grid.device).to(omega_grid.dtype)

    n_omega = len(omega_grid)
    n_sigma = len(sigma_grid)

    # Populate lookup table for different time frames.
    f_table = torch.zeros(n_sigma, n_omega, device=omega_grid.device, dtype=omega_grid.dtype)

    for eps_idx in tqdm(range(n_sigma), desc=f"Computing {base_function.__name__}"):
        f_table[eps_idx, :] = base_function(
            omega_grid,
            torch.ones_like(omega_grid) * sigma_grid[eps_idx],
            l_grid,
            tol=tol,
        )

    return f_table


def generate_igso3_lookup_table(
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Generate a lookup table for the IGSO(3) probability distribution function of angles.

    Args:
        omega_grid: Grid of angle values ranging from [0,pi] (shape is[num_omega]).
        sigma_grid: Grid of IGSO3 std dev values (shape is [num_sigma]).
        l_max: Number of terms used in the series expansion.
        tol: Small value for numerical stability.

    Returns:
        Table of function values evaluated at different angles and std dev levels. The final shape is
        [num_sigma x num_omega].
    """
    f_igso = generate_lookup_table(
        base_function=igso3_expansion,
        omega_grid=omega_grid,
        sigma_grid=sigma_grid,
        l_max=l_max,
        tol=tol,
    )
    return f_igso


def generate_dlog_igso3_lookup_table(
    omega_grid: torch.Tensor,
    sigma_grid: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Generate a lookup table for the derivative of the logarithm of the angular IGSO(3) probability
    distribution function. Used e.g. for computing scaling of SO(3) norms.

    Args:
        omega_grid: Grid of angle values ranging from [0,pi] (shape is[num_omega]).
        sigma_grid: Grid of IGSO3 std dev values (shape is [num_sigma]).
        l_max: Number of terms used in the series expansion.
        tol: Small value for numerical stability.

    Returns:
        Table of function values evaluated at different angles and std dev levels. The final shape is
        [num_sigma x num_omega].
    """
    dlog_igso = generate_lookup_table(
        base_function=dlog_igso3_expansion,
        omega_grid=omega_grid,
        sigma_grid=sigma_grid,
        l_max=l_max,
        tol=tol,
    )
    return dlog_igso




### from https://github1s.com/blt2114/twisted_diffusion_sampler/blob/main/protein_exp/data/so3_utils.py

# hat map from vector space R^3 to Lie algebra so(3)
def hat(v):
    """
    v: [..., 3]
    hat_v: [..., 3, 3]
    """
    hat_v = torch.zeros([*v.shape[:-1], 3, 3], dtype=v.dtype, device=v.device)
    hat_v[..., 0, 1], hat_v[..., 0, 2], hat_v[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return hat_v + -hat_v.transpose(-1, -2)


# vee map from Lie algebra so(3) to the vector space R^3
def vee(A):
    if not torch.allclose(A, -A.transpose(-1, -2), atol=1e-4, rtol=1e-4):
        print("Input A must be skew symmetric, Err" + str(((A - A.transpose(-1,
            -2))**2).sum(dim=[-1, -2])))
    vee_A = torch.stack([-A[..., 1, 2], A[..., 0, 2], -A[..., 0, 1]], dim=-1)
    return vee_A


# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
def Log(R):
    shape = list(R.shape[:-2])
    R_ = R.reshape([-1, 3, 3]).to(torch.float64)
    Log_R_ = rotation_vector_from_matrix(R_)
    return Log_R_.reshape(shape + [3]).to(R.dtype)


# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R): return hat(Log(R))


# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.matrix_exp(A)


# Exponential map from R^3 to SO(3)
def Exp(A): return exp(hat(A))


# Angle of rotation SO(3) to R^+, this is the norm in our chosen orthonormal basis
def Omega(R, eps=1e-6):
    # multiplying by (1-epsilon) prevents instability of arccos when provided with -1 or 1 as input.
    R_ = R.to(torch.float64)
    assert not torch.any(torch.abs(R) > 1.1)
    trace = torch.diagonal(R_, dim1=-2, dim2=-1).sum(dim=-1) * (1-eps)
    out = (trace - 1.)/2.
    out = torch.clamp(out, min=-0.9999, max=0.9999)
    return torch.arccos(out).to(R.dtype)


# exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent=None, skew_sym=None):
    assert (tangent is not None) ^ (skew_sym is not None)
    if skew_sym is None:
        skew_sym = torch.einsum('...ij,...ik->...jk', R0, tangent)
        if not  torch.allclose(skew_sym, -skew_sym.transpose(-1, -2), atol=1e-3,
                rtol=1e-3):
            print("in expmap, R0.T @ tangent must be skew symmetric")
        skew_sym = (skew_sym - torch.transpose(skew_sym, -2, -1))/2.
    exp_skew_sym = exp(skew_sym)
    return torch.einsum('...ij,...jk->...ik', R0, exp_skew_sym)


# Logarithmic map from SO(3) to tangent space at R0
def logmap(R0, R):
    R0_transpose_R = torch.einsum('...ij,...ik->...jk', R0, R)
    return torch.einsum(
        '...ij,...jk->...ik',
        R0, log(R0_transpose_R)
    )


def sample_uniform(N, M=1000):
    omega_grid = np.linspace(0, np.pi, M)
    cdf = np.cumsum(np.pi**-1 * (1-np.cos(omega_grid)), 0)/(M/np.pi)
    omegas = np.interp(np.random.rand(N), cdf, omega_grid)
    axes = np.random.randn(N, 3)
    axes = omegas[..., None]* axes/np.linalg.norm(axes, axis=-1, keepdims=True)
    axes_ = axes.reshape([-1, 3])
    Rs = exp(hat(torch.tensor(axes_)))
    Rs = Rs.reshape([N, 3, 3])
    return Rs


def sample_wrapped_normal(N, scale=1.):
    v = scale * torch.randn([N, 3])
    Rs = exp(hat(v))
    Rs = Rs.reshape([N, 3, 3])
    return Rs


def rotation_vector_from_matrix(rot_mat):
    """Convert rotation matrix (in 3D) to rotation vector (axis-angle).

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884

    Get the angle through the trace of the rotation matrix:
    The eigenvalues are:
    :math:`\{1, \cos(angle) + i \sin(angle), \cos(angle) - i \sin(angle)\}`
    so that:
    :math:`trace = 1 + 2 \cos(angle), \{-1 \leq trace \leq 3\}`
    The rotation vector is the vector associated to the skew-symmetric
    matrix
    :math:`S_r = \frac{angle}{(2 * \sin(angle) ) (R - R^T)}`

    For the edge case where the angle is close to pi,
    the rotation vector (up to sign) is derived by using the following
    equality (see the Axis-angle representation on Wikipedia):
    :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
    In nD, the rotation vector stores the :math:`n(n-1)/2` values
    of the skew-symmetric matrix representing the rotation.

    Parameters
    ----------
    rot_mat : array-like, shape=[..., n, n]
        Rotation matrix.

    Returns
    -------
    regularized_rot_vec : array-like, shape=[..., 3]
        Rotation vector.
    """
    angle = Omega(rot_mat)
    assert len(angle.shape)==1, "cannot handle vectorized Log map here"
    n_rot_mats = len(angle)
    rot_mat_transpose = torch.transpose(rot_mat, -2, -1)
    rot_vec_not_pi = vee(rot_mat - rot_mat_transpose)
    mask_0 = torch.isclose(angle, torch.tensor(0.0, dtype=angle.dtype, device=angle.device)).to(angle.dtype)
    mask_pi = torch.isclose(angle, torch_pi.to(angle.dtype), atol=1e-2).to(angle.dtype)
    mask_else = (1 - mask_0) * (1 - mask_pi)

    numerator = 0.5 * mask_0 + angle * mask_else
    denominator = (
        (1 - angle**2 / 6) * mask_0 + 2 * torch.sin(angle) * mask_else + mask_pi
    )

    rot_vec_not_pi = rot_vec_not_pi * numerator[..., None] / denominator[..., None]

    vector_outer = 0.5 * (torch.eye(3, dtype=rot_mat.dtype, device=rot_mat.device) + rot_mat)
    vector_outer = vector_outer + (torch.maximum(torch.tensor(0.0,
        dtype=vector_outer.dtype, device=rot_mat.device), vector_outer) - vector_outer)*torch.eye(3,
                dtype=vector_outer.dtype, device=rot_mat.device)
    squared_diag_comp = torch.diagonal(vector_outer, dim1=-2, dim2=-1)
    diag_comp = torch.sqrt(squared_diag_comp)
    norm_line = torch.linalg.norm(vector_outer, dim=-1)
    max_line_index = torch.argmax(norm_line, dim=-1)
    selected_line = vector_outer[range(n_rot_mats), max_line_index]
    # want
    signs = torch.sign(selected_line)
    rot_vec_pi = angle[..., None] * signs * diag_comp

    rot_vec = rot_vec_not_pi + mask_pi[..., None] * rot_vec_pi
    return regularize(rot_vec)


torch_pi = torch.tensor(3.1415926535)


def regularize(point):
    """Regularize a point to be in accordance with convention.
    In 3D, regularize the norm of the rotation vector,
    to be between 0 and pi, following the axis-angle
    representation's convention.
    If the angle is between pi and 2pi,
    the function computes its complementary in 2pi and
    inverts the direction of the rotation axis.
    Parameters

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884
    ----------
    point : array-like, shape=[...,3]
        Point.
    Returns
    -------
    regularized_point : array-like, shape=[..., 3]
        Regularized point.
    """
    theta = torch.linalg.norm(point, axis=-1)
    k = torch.floor(theta / 2.0 / torch_pi)

    # angle in [0;2pi)
    angle = theta - 2 * k * torch_pi

    # this avoids dividing by 0
    theta_eps = torch.where(torch.isclose(theta, torch.tensor(0.0,
        dtype=theta.dtype)), 1.0, theta)

    # angle in [0, pi]
    normalized_angle = torch.where(angle <= torch_pi, angle, 2 * torch_pi - angle)
    norm_ratio = torch.where(torch.isclose(theta, torch.tensor(0.0,
        dtype=theta.dtype)), 1.0, normalized_angle / theta_eps)

    # reverse sign if angle was greater than pi
    norm_ratio = torch.where(angle > torch_pi, -norm_ratio, norm_ratio)
    return torch.einsum("...,...i->...i", norm_ratio, point)
