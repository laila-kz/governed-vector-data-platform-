from ingestion.sanitization import (
    REDACTED_EMAIL,
    REDACTED_SECRET,
    sanitize_document,
    sanitize_text,
)


def test_sanitizes_email_with_dedicated_mask() -> None:
    result = sanitize_text("Contact analyst@example.com for details.")

    assert result.sanitized_text == f"Contact {REDACTED_EMAIL} for details."
    assert result.pii_masked_flag is True
    assert result.redaction_counts == {"email": 1}


def test_sanitizes_secrets_ipv4_and_national_id() -> None:
    text = (
        "api_key=abc1234567890, host 192.168.1.10, "
        "national_id: AB1234567, and SSN 123-45-6789"
    )

    result = sanitize_text(text)

    assert REDACTED_SECRET in result.sanitized_text
    assert "192.168.1.10" not in result.sanitized_text
    assert "AB1234567" not in result.sanitized_text
    assert "123-45-6789" not in result.sanitized_text
    assert result.pii_masked_flag is True
    assert result.redaction_counts == {
        "api_key": 1,
        "ipv4": 1,
        "national_id": 2,
    }


def test_clean_text_and_document_report_false_flag() -> None:
    result = sanitize_text("No sensitive values here.")
    document = sanitize_document({"title": "Title", "text": result.sanitized_text})

    assert result.pii_masked_flag is False
    assert result.redaction_counts == {}
    assert document["pii_masked_flag"] is False
    assert document["redaction_counts"] == {}