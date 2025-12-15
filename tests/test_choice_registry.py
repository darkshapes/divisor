# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from divisor.cli_menu import _CHOICE_REGISTRY


def test_registry_is_complete():
    """All keys that the original UI offered (including the empty key for Enter)"""
    expected_keys = {"", "p", "g", "j", "a", "d", "e", "l", "b", "v", "s", "w", "r"}
    assert set(_CHOICE_REGISTRY.keys()) == expected_keys
