"""
tests/test_auth.py

Regression test for the consent gate added to sign-up: a coach must
explicitly confirm the data-use notice (including parent/guardian consent
for athletes under 18) before an account can be created. This must be
rejected before ever reaching the network — no Supabase credentials
should be required for this test to pass.
"""

import auth


class TestSignUpConsentGate:
    def test_missing_consent_is_rejected_before_network_call(self):
        result = auth.sign_up("coach@example.com", "password123", consent_given=False)
        assert result["status"] == "error"
        assert "data-use notice" in result["message"].lower()

    def test_default_consent_value_is_false(self):
        """consent_given must default to False — an account can never be
        created by accident just by omitting the argument."""
        result = auth.sign_up("coach@example.com", "password123")
        assert result["status"] == "error"

    def test_invalid_email_is_rejected_even_with_consent(self):
        """Existing validation must still run regardless of consent."""
        result = auth.sign_up("not-an-email", "password123", consent_given=True)
        assert result["status"] == "error"
        assert "email" in result["message"].lower()

    def test_short_password_is_rejected_even_with_consent(self):
        result = auth.sign_up("coach@example.com", "short", consent_given=True)
        assert result["status"] == "error"
        assert "password" in result["message"].lower()
