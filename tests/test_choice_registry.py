# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from divisor.cli_menu import _CHOICE_REGISTRY


def test_registry_is_complete():
    """All keys that the original UI offered (including the empty key for Enter)"""
    expected_keys = {"", "g", "s", "r", "l", "b", "a", "v", "d", "e", "p", "j"}
    assert set(_CHOICE_REGISTRY.keys()) == expected_keys
