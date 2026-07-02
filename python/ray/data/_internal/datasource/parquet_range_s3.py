"""AWS S3 object identity support for Parquet range reads.

This module is intentionally independent of the Parquet datasource adapters.
It handles only ordinary ``s3://bucket/key`` URIs authenticated through the
ambient AWS credential chain.  In particular, identity tokens never contain a
URI, filesystem object, endpoint, or credentials.

``s3fs`` stays an optional, lazy import.  Callers can therefore use the pure
URI and identity-codec helpers without installing the S3 dependency.
"""

from __future__ import annotations

import base64
import binascii
import json
import numbers
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import quote, unquote, urlsplit


_S3_IDENTITY_VERSION = "s3-v1"
_MAX_IDENTITY_TOKEN_BYTES = 16 * 1024
_MAX_IDENTITY_FIELD_LENGTH = 4096
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


class S3SourceIdentityError(RuntimeError):
    """An S3 URI or source identity is unsafe or could not be verified."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class S3ObjectLocation:
    """A credential-free, decoded S3 bucket and object key."""

    bucket: str
    key: str

    @property
    def s3fs_path(self) -> str:
        """Return the protocol-free path expected by ``s3fs``."""

        return f"{self.bucket}/{self.key}"

    @property
    def uri(self) -> str:
        """Return a canonical URI, escaping URI delimiters inside the key."""

        encoded_key = quote(self.key, safe="/!$&'()*+,-.:;=@_~")
        return f"s3://{self.bucket}/{encoded_key}"


def parse_s3_uri(value: Any) -> S3ObjectLocation:
    """Parse a plain, credential-free ``s3://bucket/key`` URI.

    URI percent escapes are decoded exactly once before the path is passed to
    ``s3fs``.  Queries, fragments, alternate schemes, embedded credentials,
    and bucket-root URIs are rejected rather than being reinterpreted.
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise S3SourceIdentityError(
            "An S3 source must be a non-empty string URI.",
            reason_code="invalid_s3_uri",
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise S3SourceIdentityError(
            "An S3 source URI cannot contain control characters.",
            reason_code="invalid_s3_uri",
        )

    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise S3SourceIdentityError(
            "The S3 source URI is malformed.",
            reason_code="invalid_s3_uri",
        ) from exc

    if parsed.scheme != "s3":
        raise S3SourceIdentityError(
            "Only plain s3:// source URIs are supported.",
            reason_code="unsupported_s3_scheme",
        )
    # Check the raw authority before touching ``hostname``.  That avoids
    # accepting or reflecting a userinfo component in an error message.
    if "@" in parsed.netloc:
        raise S3SourceIdentityError(
            "Credentials embedded in an S3 URI are not supported.",
            reason_code="embedded_s3_credentials",
        )
    if not parsed.netloc or ":" in parsed.netloc:
        raise S3SourceIdentityError(
            "An S3 source URI must contain one bucket name and no port.",
            reason_code="invalid_s3_uri",
        )
    if parsed.query or parsed.fragment:
        raise S3SourceIdentityError(
            "S3 source URI queries and fragments are not supported.",
            reason_code="unsupported_s3_uri_components",
        )
    if not parsed.path.startswith("/") or len(parsed.path) == 1:
        raise S3SourceIdentityError(
            "An S3 source URI must identify an object key.",
            reason_code="invalid_s3_uri",
        )
    if _INVALID_PERCENT_ESCAPE.search(parsed.path):
        raise S3SourceIdentityError(
            "The S3 source URI contains an invalid percent escape.",
            reason_code="invalid_s3_uri",
        )

    try:
        key = unquote(parsed.path[1:], encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise S3SourceIdentityError(
            "The S3 object key is not valid UTF-8.",
            reason_code="invalid_s3_uri",
        ) from exc
    bucket = parsed.netloc.lower()
    if (
        not key
        or any(ord(char) < 32 or ord(char) == 127 for char in key)
        or any(char.isspace() for char in bucket)
    ):
        raise S3SourceIdentityError(
            "The S3 source URI contains an invalid bucket or object key.",
            reason_code="invalid_s3_uri",
        )
    return S3ObjectLocation(bucket=bucket, key=key)


def _validate_identity_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise S3SourceIdentityError(
            f"S3 source metadata has no valid {field}.",
            reason_code="invalid_s3_metadata",
        )
    if len(value) > _MAX_IDENTITY_FIELD_LENGTH or any(
        ord(char) < 32 or ord(char) == 127 for char in value
    ):
        raise S3SourceIdentityError(
            f"S3 source metadata contains an invalid {field}.",
            reason_code="invalid_s3_metadata",
        )
    return value


def _normalize_etag(value: Any) -> str:
    etag = _validate_identity_text(value, field="ETag").strip()
    if etag.startswith("W/") or etag.startswith("w/"):
        raise S3SourceIdentityError(
            "A weak ETag cannot identify an S3 source object.",
            reason_code="invalid_s3_metadata",
        )
    if etag.startswith('"') or etag.endswith('"'):
        if len(etag) < 2 or not (etag.startswith('"') and etag.endswith('"')):
            raise S3SourceIdentityError(
                "S3 source metadata contains a malformed ETag.",
                reason_code="invalid_s3_metadata",
            )
        etag = etag[1:-1]
    if '"' in etag:
        raise S3SourceIdentityError(
            "S3 source metadata contains a malformed ETag.",
            reason_code="invalid_s3_metadata",
        )
    return _validate_identity_text(etag, field="ETag")


def _normalize_last_modified(value: Any) -> str:
    timestamp: Optional[datetime] = None
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str) and value:
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                timestamp = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                timestamp = None
    if timestamp is None or timestamp.tzinfo is None:
        raise S3SourceIdentityError(
            "S3 source metadata has no timezone-aware LastModified value.",
            reason_code="invalid_s3_metadata",
        )
    return timestamp.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _metadata_value(
    metadata: Mapping[str, Any], names: Tuple[str, ...]
) -> Tuple[bool, Any]:
    for name in names:
        if name in metadata:
            return True, metadata[name]
    return False, None


@dataclass(frozen=True)
class S3SourceIdentity:
    """Stable HeadObject fields captured alongside an S3 Parquet footer.

    S3 always supplies size and ETag.  A concrete VersionId is the strongest
    mutation discriminator.  For an unversioned object, LastModified is used
    instead.  Exactly one of those two fields is stored.
    """

    size: int
    etag: str
    version_id: Optional[str] = None
    last_modified: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, numbers.Integral)
            or self.size < 0
        ):
            raise S3SourceIdentityError(
                "An S3 source identity contains an invalid size.",
                reason_code="invalid_source_identity",
            )
        try:
            normalized_etag = _normalize_etag(self.etag)
        except S3SourceIdentityError as exc:
            raise S3SourceIdentityError(
                "An S3 source identity contains an invalid ETag.",
                reason_code="invalid_source_identity",
            ) from exc
        if normalized_etag != self.etag:
            raise S3SourceIdentityError(
                "An S3 source identity contains a non-canonical ETag.",
                reason_code="invalid_source_identity",
            )
        if (self.version_id is None) == (self.last_modified is None):
            raise S3SourceIdentityError(
                "An S3 source identity must contain exactly one mutation marker.",
                reason_code="invalid_source_identity",
            )
        marker = self.version_id or self.last_modified
        if not isinstance(marker, str) or not marker or len(marker) > 4096:
            raise S3SourceIdentityError(
                "An S3 source identity contains an invalid mutation marker.",
                reason_code="invalid_source_identity",
            )
        if any(ord(char) < 32 or ord(char) == 127 for char in marker):
            raise S3SourceIdentityError(
                "An S3 source identity contains an invalid mutation marker.",
                reason_code="invalid_source_identity",
            )
        if self.last_modified is not None:
            try:
                normalized = _normalize_last_modified(self.last_modified)
            except S3SourceIdentityError as exc:
                raise S3SourceIdentityError(
                    "An S3 source identity contains an invalid LastModified value.",
                    reason_code="invalid_source_identity",
                ) from exc
            if normalized != self.last_modified:
                raise S3SourceIdentityError(
                    "An S3 source identity has a non-canonical LastModified value.",
                    reason_code="invalid_source_identity",
                )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "S3SourceIdentity":
        """Build an identity from an ``s3fs.info``/HeadObject response."""

        if not isinstance(metadata, Mapping):
            raise S3SourceIdentityError(
                "S3 source metadata must be a mapping.",
                reason_code="invalid_s3_metadata",
            )
        found_size, raw_size = _metadata_value(metadata, ("size", "Size"))
        if (
            not found_size
            or isinstance(raw_size, bool)
            or not isinstance(raw_size, numbers.Integral)
            or raw_size < 0
        ):
            raise S3SourceIdentityError(
                "S3 source metadata contains an invalid object size.",
                reason_code="invalid_s3_metadata",
            )
        found_etag, raw_etag = _metadata_value(metadata, ("ETag", "etag"))
        if not found_etag:
            raise S3SourceIdentityError(
                "S3 source metadata contains no ETag.",
                reason_code="invalid_s3_metadata",
            )
        etag = _normalize_etag(raw_etag)

        found_version, raw_version = _metadata_value(
            metadata, ("VersionId", "version_id")
        )
        if found_version and raw_version not in (None, ""):
            version_id = _validate_identity_text(
                raw_version, field="VersionId"
            )
            return cls(size=int(raw_size), etag=etag, version_id=version_id)

        found_modified, raw_modified = _metadata_value(
            metadata, ("LastModified", "last_modified")
        )
        if not found_modified:
            raise S3SourceIdentityError(
                "Unversioned S3 source metadata contains no LastModified value.",
                reason_code="invalid_s3_metadata",
            )
        return cls(
            size=int(raw_size),
            etag=etag,
            last_modified=_normalize_last_modified(raw_modified),
        )

    def encode(self) -> str:
        payload = {"etag": self.etag, "size": int(self.size)}
        if self.version_id is not None:
            payload["version_id"] = self.version_id
        else:
            payload["last_modified"] = self.last_modified
        serialized = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        encoded = base64.urlsafe_b64encode(serialized).rstrip(b"=").decode("ascii")
        return f"{_S3_IDENTITY_VERSION}:{encoded}"

    @classmethod
    def decode(cls, value: str) -> "S3SourceIdentity":
        if not isinstance(value, str):
            raise S3SourceIdentityError(
                "An S3 source identity must be a string.",
                reason_code="invalid_source_identity",
            )
        if len(value.encode("utf-8")) > _MAX_IDENTITY_TOKEN_BYTES:
            raise S3SourceIdentityError(
                "The S3 source identity is too large.",
                reason_code="invalid_source_identity",
            )
        prefix, separator, encoded = value.partition(":")
        if prefix != _S3_IDENTITY_VERSION or not separator or not encoded:
            raise S3SourceIdentityError(
                "The S3 source identity has an unsupported format.",
                reason_code="invalid_source_identity",
            )
        try:
            padding = "=" * (-len(encoded) % 4)
            serialized = base64.b64decode(
                encoded + padding, altchars=b"-_", validate=True
            )
            payload = json.loads(serialized.decode("utf-8"))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise S3SourceIdentityError(
                "The S3 source identity payload is malformed.",
                reason_code="invalid_source_identity",
            ) from exc
        if not isinstance(payload, dict):
            raise S3SourceIdentityError(
                "The S3 source identity payload must be an object.",
                reason_code="invalid_source_identity",
            )
        common = {"size", "etag"}
        keys = set(payload)
        if keys == common | {"version_id"}:
            identity = cls(
                size=payload["size"],
                etag=payload["etag"],
                version_id=payload["version_id"],
            )
        elif keys == common | {"last_modified"}:
            identity = cls(
                size=payload["size"],
                etag=payload["etag"],
                last_modified=payload["last_modified"],
            )
        else:
            raise S3SourceIdentityError(
                "The S3 source identity contains unexpected fields.",
                reason_code="invalid_source_identity",
            )
        # Accept only the output of this version's encoder.  This rules out
        # duplicate JSON keys, alternate number spellings, gratuitous padding,
        # and other ambiguous representations of the same identity.
        if identity.encode() != value:
            raise S3SourceIdentityError(
                "The S3 source identity has a non-canonical encoding.",
                reason_code="invalid_source_identity",
            )
        return identity


@lru_cache(maxsize=1)
def _create_ambient_s3_filesystem() -> Any:
    """Create s3fs with only its ambient AWS credential-provider chain."""

    try:
        import s3fs
    except ImportError as exc:
        raise S3SourceIdentityError(
            "S3 range planning requires the optional 's3fs' dependency.",
            reason_code="s3fs_dependency_missing",
        ) from exc
    try:
        return s3fs.S3FileSystem(anon=False)
    except Exception as exc:
        raise S3SourceIdentityError(
            "Unable to initialize the ambient-IAM S3 filesystem.",
            reason_code="s3_filesystem_initialization_failed",
        ) from exc


def _read_s3_source_identity(
    location: S3ObjectLocation, filesystem: Any
) -> S3SourceIdentity:
    path = location.s3fs_path
    try:
        invalidate_cache = getattr(filesystem, "invalidate_cache", None)
        if callable(invalidate_cache):
            invalidate_cache(path)
        metadata = filesystem.info(path)
    except Exception as exc:
        raise S3SourceIdentityError(
            f"Unable to inspect S3 source {location.uri!r}.",
            reason_code="source_unavailable",
        ) from exc
    return S3SourceIdentity.from_metadata(metadata)


def capture_s3_source_identity(uri: Any, *, filesystem: Any = None) -> str:
    """Capture a compact, credential-free identity for one S3 object."""

    location = parse_s3_uri(uri)
    if filesystem is None:
        filesystem = _create_ambient_s3_filesystem()
    return _read_s3_source_identity(location, filesystem).encode()


def verify_s3_source_identity(
    uri: Any, expected: str, *, filesystem: Any = None
) -> bool:
    """Verify that the current S3 object is the one observed during planning."""

    location = parse_s3_uri(uri)
    expected_identity = S3SourceIdentity.decode(expected)
    if filesystem is None:
        filesystem = _create_ambient_s3_filesystem()
    current_identity = _read_s3_source_identity(location, filesystem)
    if current_identity != expected_identity:
        raise S3SourceIdentityError(
            f"S3 source {location.uri!r} changed after planning.",
            reason_code="source_changed",
        )
    return True
