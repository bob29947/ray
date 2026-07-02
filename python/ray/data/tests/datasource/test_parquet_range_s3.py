import base64
import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ray.data._internal.datasource import parquet_range_s3
from ray.data._internal.datasource.parquet_range_s3 import (
    S3ObjectLocation,
    S3SourceIdentity,
    S3SourceIdentityError,
    capture_s3_source_identity,
    parse_s3_uri,
    verify_s3_source_identity,
)


@pytest.fixture(autouse=True)
def clear_ambient_s3_filesystem_cache():
    factory = parquet_range_s3._create_ambient_s3_filesystem
    factory.cache_clear()
    yield
    factory.cache_clear()


class FakeS3FileSystem:
    def __init__(self, metadata=None, error=None):
        self.metadata = metadata
        self.error = error
        self.invalidated = []
        self.info_calls = []
        # These model sensitive state commonly held by a real filesystem.  It
        # must never appear in an encoded source identity.
        self.key = "AKIA_NOT_SERIALIZED"
        self.secret = "SECRET_NOT_SERIALIZED"

    def invalidate_cache(self, path):
        self.invalidated.append(path)

    def info(self, path):
        self.info_calls.append(path)
        if self.error is not None:
            raise self.error
        return self.metadata


def _unversioned_metadata(**updates):
    metadata = {
        "Size": 123,
        "ETag": '"0123456789abcdef-2"',
        "LastModified": datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc),
    }
    metadata.update(updates)
    return metadata


def test_parse_s3_uri_and_canonicalize_escaped_key():
    location = parse_s3_uri(
        "s3://My-Bucket/folder/a%20b%23c%3F100%25%2B.parquet"
    )

    assert location == S3ObjectLocation(
        bucket="my-bucket", key="folder/a b#c?100%+.parquet"
    )
    assert location.s3fs_path == "my-bucket/folder/a b#c?100%+.parquet"
    assert (
        location.uri
        == "s3://my-bucket/folder/a%20b%23c%3F100%25+.parquet"
    )
    assert parse_s3_uri(location.uri) == location


@pytest.mark.parametrize(
    ("uri", "reason_code"),
    [
        (None, "invalid_s3_uri"),
        (" s3://bucket/key", "invalid_s3_uri"),
        ("https://bucket/key", "unsupported_s3_scheme"),
        ("s3a://bucket/key", "unsupported_s3_scheme"),
        ("s3://access:secret@bucket/key", "embedded_s3_credentials"),
        ("s3://bucket:443/key", "invalid_s3_uri"),
        ("s3://bucket", "invalid_s3_uri"),
        ("s3://bucket/", "invalid_s3_uri"),
        ("s3://bucket/key?versionId=1", "unsupported_s3_uri_components"),
        ("s3://bucket/key#fragment", "unsupported_s3_uri_components"),
        ("s3://bucket/bad%2", "invalid_s3_uri"),
        ("s3://bucket/bad%FF", "invalid_s3_uri"),
        ("s3://bucket/control%00", "invalid_s3_uri"),
    ],
)
def test_parse_s3_uri_fails_closed(uri, reason_code):
    with pytest.raises(S3SourceIdentityError) as exc_info:
        parse_s3_uri(uri)

    assert exc_info.value.reason_code == reason_code
    if isinstance(uri, str) and "secret" in uri:
        assert "secret" not in str(exc_info.value)


def test_versioned_metadata_uses_size_etag_and_version_only():
    identity = S3SourceIdentity.from_metadata(
        {
            "size": 12,
            "etag": '  "ABCDEF-3"  ',
            "version_id": "version-7",
            # VersionId is stronger and makes this field irrelevant.
            "last_modified": "not-a-timestamp",
        }
    )

    assert identity == S3SourceIdentity(
        size=12, etag="ABCDEF-3", version_id="version-7"
    )
    assert identity.last_modified is None


def test_unversioned_metadata_normalizes_last_modified_to_utc():
    offset = timezone(timedelta(hours=-7))
    identity = S3SourceIdentity.from_metadata(
        _unversioned_metadata(
            LastModified=datetime(2026, 7, 1, 5, 30, 1, 2, tzinfo=offset)
        )
    )
    from_http_date = S3SourceIdentity.from_metadata(
        _unversioned_metadata(LastModified="Wed, 01 Jul 2026 12:30:00 GMT")
    )

    assert identity.last_modified == "2026-07-01T12:30:01.000002Z"
    assert from_http_date.last_modified == "2026-07-01T12:30:00.000000Z"


@pytest.mark.parametrize("version_id", [None, ""])
def test_missing_version_id_falls_back_to_last_modified(version_id):
    identity = S3SourceIdentity.from_metadata(
        _unversioned_metadata(VersionId=version_id)
    )

    assert identity.version_id is None
    assert identity.last_modified == "2026-07-01T12:30:00.000000Z"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"size": True, "ETag": '"abc"', "VersionId": "v1"},
        {"size": -1, "ETag": '"abc"', "VersionId": "v1"},
        {"size": 1, "VersionId": "v1"},
        {"size": 1, "ETag": "", "VersionId": "v1"},
        {"size": 1, "ETag": 'W/"abc"', "VersionId": "v1"},
        {"size": 1, "ETag": '"abc', "VersionId": "v1"},
        {"size": 1, "ETag": '"abc"'},
        {"size": 1, "ETag": '"abc"', "LastModified": "naive"},
    ],
)
def test_invalid_head_metadata_has_explicit_reason(metadata):
    with pytest.raises(S3SourceIdentityError) as exc_info:
        S3SourceIdentity.from_metadata(metadata)

    assert exc_info.value.reason_code == "invalid_s3_metadata"


@pytest.mark.parametrize("version_id", ["null", "version/with+symbols="])
def test_concrete_version_ids_do_not_require_last_modified(version_id):
    identity = S3SourceIdentity.from_metadata(
        {"Size": 0, "ETag": '"abc"', "VersionId": version_id}
    )

    assert identity.version_id == version_id
    assert identity.last_modified is None


def test_identity_codec_is_deterministic_safe_and_pickle_serializable():
    identity = S3SourceIdentity(
        size=123,
        etag="ABCDEF-3",
        version_id="version:with/slashes+and=padding",
    )

    encoded = identity.encode()
    restored = S3SourceIdentity.decode(encoded)

    assert encoded.startswith("s3-v1:")
    assert restored == identity
    assert restored.encode() == encoded
    assert pickle.loads(pickle.dumps(restored)) == identity
    assert "version:with/slashes" not in encoded


def test_identity_decoder_rejects_padded_or_noncanonical_json():
    identity = S3SourceIdentity(size=1, etag="abc", version_id="v")
    padded = identity.encode() + "="
    noncanonical = _encode_payload(
        {"version_id": "v", "etag": "abc", "size": 1}
    )

    for value in (padded, noncanonical):
        with pytest.raises(S3SourceIdentityError) as exc_info:
            S3SourceIdentity.decode(value)
        assert exc_info.value.reason_code == "invalid_source_identity"


def _encode_payload(payload):
    serialized = json.dumps(payload, separators=(",", ":")).encode()
    token = base64.urlsafe_b64encode(serialized).rstrip(b"=").decode()
    return f"s3-v1:{token}"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "posix-v1:1:2:3:4",
        "s3-v1:not+base64!",
        _encode_payload([]),
        _encode_payload({"size": 1, "etag": "abc"}),
        _encode_payload(
            {"size": 1, "etag": "abc", "version_id": "v", "extra": 1}
        ),
        _encode_payload({"size": True, "etag": "abc", "version_id": "v"}),
        _encode_payload(
            {
                "size": 1,
                "etag": '"noncanonical"',
                "version_id": "v",
            }
        ),
    ],
)
def test_identity_decoder_rejects_malformed_or_ambiguous_values(value):
    with pytest.raises(S3SourceIdentityError) as exc_info:
        S3SourceIdentity.decode(value)

    assert exc_info.value.reason_code == "invalid_source_identity"


def test_capture_and_verify_use_fresh_protocol_free_head_and_no_credentials():
    filesystem = FakeS3FileSystem(_unversioned_metadata())

    encoded = capture_s3_source_identity(
        "s3://bucket/a%20b.parquet", filesystem=filesystem
    )

    assert filesystem.invalidated == ["bucket/a b.parquet"]
    assert filesystem.info_calls == ["bucket/a b.parquet"]
    assert "bucket" not in encoded
    assert filesystem.key not in encoded
    assert filesystem.secret not in encoded
    payload_token = encoded.partition(":")[2]
    payload_token += "=" * (-len(payload_token) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_token))
    assert set(payload) == {"etag", "last_modified", "size"}
    assert verify_s3_source_identity(
        "s3://bucket/a%20b.parquet", encoded, filesystem=filesystem
    )
    assert filesystem.invalidated == ["bucket/a b.parquet"] * 2
    assert filesystem.info_calls == ["bucket/a b.parquet"] * 2


def test_verify_detects_each_object_identity_change():
    uri = "s3://bucket/data.parquet"
    original = capture_s3_source_identity(
        uri, filesystem=FakeS3FileSystem(_unversioned_metadata())
    )

    for update in (
        {"Size": 124},
        {"ETag": '"changed"'},
        {"LastModified": datetime(2026, 7, 1, 12, 31, tzinfo=timezone.utc)},
    ):
        with pytest.raises(S3SourceIdentityError) as exc_info:
            verify_s3_source_identity(
                uri,
                original,
                filesystem=FakeS3FileSystem(_unversioned_metadata(**update)),
            )
        assert exc_info.value.reason_code == "source_changed"


def test_verify_versioned_object_detects_version_change():
    uri = "s3://bucket/data.parquet"
    original_metadata = {"Size": 1, "ETag": '"abc"', "VersionId": "v1"}
    original = capture_s3_source_identity(
        uri, filesystem=FakeS3FileSystem(original_metadata)
    )
    changed = dict(original_metadata, VersionId="v2")

    with pytest.raises(S3SourceIdentityError) as exc_info:
        verify_s3_source_identity(
            uri, original, filesystem=FakeS3FileSystem(changed)
        )

    assert exc_info.value.reason_code == "source_changed"


def test_source_head_failure_has_explicit_reason_and_sanitized_location():
    filesystem = FakeS3FileSystem(error=FileNotFoundError("backend secret"))

    with pytest.raises(S3SourceIdentityError) as exc_info:
        capture_s3_source_identity(
            "s3://bucket/missing.parquet", filesystem=filesystem
        )

    assert exc_info.value.reason_code == "source_unavailable"
    assert "backend secret" not in str(exc_info.value)


def test_expected_identity_is_validated_before_ambient_filesystem_creation(
    monkeypatch,
):
    def unexpected_filesystem():
        raise AssertionError("filesystem should not be created")

    monkeypatch.setattr(
        parquet_range_s3, "_create_ambient_s3_filesystem", unexpected_filesystem
    )

    with pytest.raises(S3SourceIdentityError) as exc_info:
        verify_s3_source_identity("s3://bucket/key", "bad")

    assert exc_info.value.reason_code == "invalid_source_identity"


def test_default_filesystem_uses_only_ambient_iam(monkeypatch):
    calls = []
    filesystem = FakeS3FileSystem(_unversioned_metadata())

    def constructor(**kwargs):
        calls.append(kwargs)
        return filesystem

    monkeypatch.setitem(
        sys.modules, "s3fs", SimpleNamespace(S3FileSystem=constructor)
    )

    capture_s3_source_identity("s3://bucket/key")

    assert calls == [{"anon": False}]


def test_missing_s3fs_dependency_has_explicit_reason(monkeypatch):
    real_import = __import__

    def import_without_s3fs(name, *args, **kwargs):
        if name == "s3fs":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "s3fs", raising=False)
    monkeypatch.setattr("builtins.__import__", import_without_s3fs)

    with pytest.raises(S3SourceIdentityError) as exc_info:
        capture_s3_source_identity("s3://bucket/key")

    assert exc_info.value.reason_code == "s3fs_dependency_missing"


def test_ambient_filesystem_initialization_failure_is_sanitized(monkeypatch):
    def constructor(**kwargs):
        raise RuntimeError("credential-provider-secret")

    monkeypatch.setitem(
        sys.modules, "s3fs", SimpleNamespace(S3FileSystem=constructor)
    )

    with pytest.raises(S3SourceIdentityError) as exc_info:
        capture_s3_source_identity("s3://bucket/key")

    assert exc_info.value.reason_code == "s3_filesystem_initialization_failed"
    assert "credential-provider-secret" not in str(exc_info.value)
