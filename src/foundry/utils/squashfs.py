"""Squashfs filesystem mount management.

Provides a singleton manager for mounting and caching squashfs filesystems.
Mounts are lazy (happen on first access) and persist for the lifetime of the program.
"""

import atexit
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class SquashfsManager:
    """Singleton manager for squashfs filesystem mounts.

    Provides lazy mounting of squashfs files with automatic caching and cleanup.
    Thread-safe for use with multiprocessing/DataLoader workers.

    Examples:
        Get mount directory (mounts automatically if needed)::

            from foundry.utils.squashfs import SquashfsManager

            mount_dir = SquashfsManager.get_mount("/path/to/file.sqfs")
            file_path = os.path.join(mount_dir, "internal/path/to/file")

        Inspect active mounts::

            mounts = SquashfsManager.list_mounts()
            print(f"Active mounts: {len(mounts)}")

        Cleanup (typically for tests)::

            SquashfsManager.unmount_all()
    """

    _mounts: dict[str, str] = {}
    _lock = threading.Lock()
    _cleanup_registered = False

    @classmethod
    def get_mount(cls, sqfs_file: str) -> str:
        """Get mount directory for squashfs file, mounting if needed.

        Args:
            sqfs_file: Path to the .sqfs file.

        Returns:
            Path to the mount directory containing the squashfs contents.

        Raises:
            FileNotFoundError: If the sqfs file doesn't exist.
            RuntimeError: If mounting fails.
        """
        sqfs_file = str(Path(sqfs_file).resolve())

        # Check file exists
        if not os.path.exists(sqfs_file):
            raise FileNotFoundError(f"Squashfs file not found: {sqfs_file}")

        with cls._lock:
            # Register cleanup on first use
            if not cls._cleanup_registered:
                atexit.register(cls._cleanup_on_exit)
                cls._cleanup_registered = True

            # Return cached mount if exists
            if sqfs_file in cls._mounts:
                return cls._mounts[sqfs_file]

            # Mount and cache
            mount_dir = cls._mount(sqfs_file)
            cls._mounts[sqfs_file] = mount_dir
            logger.info(f"Mounted {sqfs_file} at {mount_dir}")
            return mount_dir

    @classmethod
    def _mount(cls, sqfs_file: str) -> str:
        """Internal: Actually perform the squashfs mount.

        Args:
            sqfs_file: Path to the .sqfs file.

        Returns:
            Path to the mount directory.

        Raises:
            RuntimeError: If mounting fails.
        """
        mount_dir = tempfile.mkdtemp(prefix="sqfs_")
        try:
            subprocess.run(
                ["squashfuse", sqfs_file, mount_dir],
                check=True,
                capture_output=True,
            )
            return mount_dir
        except subprocess.CalledProcessError as e:
            # Cleanup failed mount directory
            try:
                os.rmdir(mount_dir)
            except OSError:
                pass
            raise RuntimeError(
                f"Failed to mount {sqfs_file}: {e.stderr.decode()}"
            ) from e

    @classmethod
    def _unmount(cls, sqfs_file: str) -> None:
        """Internal: Unmount a squashfs filesystem.

        Args:
            sqfs_file: Path to the .sqfs file to unmount.
        """
        if sqfs_file not in cls._mounts:
            return

        mount_dir = cls._mounts[sqfs_file]
        try:
            subprocess.run(
                ["fusermount", "-u", mount_dir],
                check=True,
                capture_output=True,
            )
            os.rmdir(mount_dir)
            logger.info(f"Unmounted {sqfs_file}")
        except (subprocess.CalledProcessError, OSError) as e:
            logger.warning(f"Failed to unmount {sqfs_file}: {e}")
        finally:
            del cls._mounts[sqfs_file]

    @classmethod
    def list_mounts(cls) -> dict[str, str]:
        """List all active squashfs mounts.

        Returns:
            Dictionary mapping sqfs file paths to mount directories.

        Examples:
            >>> mounts = SquashfsManager.list_mounts()
            >>> print(f"Active mounts: {len(mounts)}")
            >>> for sqfs, mount_dir in mounts.items():
            ...     print(f"{sqfs} -> {mount_dir}")
        """
        with cls._lock:
            return cls._mounts.copy()

    @classmethod
    def unmount_all(cls) -> None:
        """Unmount all squashfs filesystems.

        Useful for cleanup in tests or explicit resource management.
        Normally not needed as cleanup happens automatically on program exit.

        Examples:
            >>> SquashfsManager.unmount_all()  # Clean up in test teardown
        """
        with cls._lock:
            sqfs_files = list(cls._mounts.keys())
            for sqfs_file in sqfs_files:
                cls._unmount(sqfs_file)

    @classmethod
    def _cleanup_on_exit(cls) -> None:
        """Cleanup hook called on program exit via atexit."""
        logger.debug("Cleaning up squashfs mounts on exit")
        cls.unmount_all()
