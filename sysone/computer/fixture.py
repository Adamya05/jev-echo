"""Synthetic screens.

Lets the decision layer be exercised and tested without Accessibility
permission, and without clicking anything real. Same Screen/Element
shape the live capture produces, with ref=None so execution is
impossible by construction.
"""

from __future__ import annotations

from sysone.computer.ax import Element, Screen


def _el(i: int, label: str, role: str, pos: str, *, enabled=True,
        focused=False, editable=False) -> Element:
    return Element(f"e{i}", label, role, enabled, focused, editable, pos,
                   (0, 0, 0, 0), None)


def mail_compose() -> Screen:
    labels = [
        ("To", "AXTextField", "upper left", dict(editable=True, focused=True)),
        ("Cc", "AXTextField", "upper left", dict(editable=True)),
        ("Subject", "AXTextField", "upper center", dict(editable=True)),
        ("Message body", "AXTextArea", "middle center", dict(editable=True)),
        ("Send", "AXButton", "upper right", {}),
        ("Save as Draft", "AXButton", "upper right", {}),
        ("Attach File", "AXButton", "upper center", {}),
        ("Delete Draft", "AXButton", "upper right", {}),
        ("Format", "AXPopUpButton", "upper center", {}),
        ("Close", "AXButton", "upper left", {}),
    ]
    els = [_el(i + 1, l, r, p, **kw) for i, (l, r, p, kw) in enumerate(labels)]
    return Screen("Mail", "New Message", els, 3.9)


def settings_panel() -> Screen:
    """Crowded enough to trigger tournament sampling."""
    rows = [
        "General", "Appearance", "Accessibility", "Privacy & Security", "Displays",
        "Sound", "Focus", "Screen Time", "Notifications", "Wallpaper",
        "Battery", "Lock Screen", "Touch ID & Password", "Users & Groups",
        "Keyboard", "Trackpad", "Mouse", "Printers & Scanners", "Network",
        "Bluetooth", "Wi-Fi", "Game Center", "Wallet & Apple Pay", "Internet Accounts",
        "Control Centre", "Siri & Spotlight", "Desktop & Dock", "Date & Time",
        "Sharing", "Time Machine", "Transfer or Reset", "Software Update",
        "Storage", "Energy Saver", "Login Items", "Language & Region",
    ]
    els = [_el(1, "Search", "AXSearchField", "upper left", editable=True)]
    els += [_el(i + 2, r, "AXRow", "middle left") for i, r in enumerate(rows)]
    els.append(_el(len(els) + 1, "Back", "AXButton", "upper left"))
    return Screen("System Settings", "System Settings", els, 6.2)


def finder_confirm() -> Screen:
    labels = [
        ("Are you sure you want to delete 3 items?", "AXStaticText", "middle center", {}),
        ("Delete", "AXButton", "lower right", {}),
        ("Cancel", "AXButton", "lower right", {}),
    ]
    els = [_el(i + 1, l, r, p, **kw) for i, (l, r, p, kw) in enumerate(labels)]
    return Screen("Finder", "Confirm Delete", els, 2.1)


SCREENS = {
    "mail": mail_compose,
    "settings": settings_panel,
    "confirm": finder_confirm,
}
