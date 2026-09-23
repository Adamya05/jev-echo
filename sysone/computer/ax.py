"""macOS accessibility-tree capture.

The tree, not the pixels. A window snapshot is a few milliseconds and a
few hundred tokens, against ~1.5s and a five-figure token bill for a
screenshot through a vision model. It is also more precise: roles and
enabled-state are facts the app reports, not inferences from an image.

No CoreML, no Xcode, no vision model -- just ApplicationServices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from AppKit import NSWorkspace
from ApplicationServices import (
    AXIsProcessTrusted,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    AXUIElementPerformAction,
    AXUIElementSetAttributeValue,
    AXValueGetValue,
    kAXChildrenAttribute,
    kAXDescriptionAttribute,
    kAXEnabledAttribute,
    kAXFocusedAttribute,
    kAXPositionAttribute,
    kAXPressAction,
    kAXRoleAttribute,
    kAXSizeAttribute,
    kAXSubroleAttribute,
    kAXTitleAttribute,
    kAXValueAttribute,
    kAXValueTypeCGPoint,
    kAXValueTypeCGSize,
    kAXWindowsAttribute,
)

# Roles worth showing the model. Everything else is layout scaffolding.
ACTIONABLE = {
    "AXButton", "AXTextField", "AXTextArea", "AXCheckBox", "AXRadioButton",
    "AXPopUpButton", "AXMenuItem", "AXMenuButton", "AXLink", "AXComboBox",
    "AXSlider", "AXIncrementor", "AXDisclosureTriangle", "AXCell", "AXRow",
    "AXTabButton", "AXSearchField", "AXStaticText",
}
EDITABLE = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}


@dataclass
class Element:
    """One UI element. `ref` never leaves the machine."""
    id: str
    label: str
    role: str
    enabled: bool
    focused: bool
    editable: bool
    position: str          # coarse, e.g. "upper left"
    rect: tuple[float, float, float, float]
    ref: Any               # live AXUIElement handle -- local only

    def for_model(self) -> dict[str, Any]:
        """The projection that is safe and useful to send over the wire."""
        d = {"id": self.id, "label": self.label, "role": self.role.removeprefix("AX")}
        if not self.enabled:
            d["enabled"] = False
        if self.focused:
            d["focused"] = True
        if self.editable:
            d["editable"] = True
        d["position"] = self.position
        return d

    def describe(self) -> str:
        flags = "".join([
            "" if self.enabled else " [disabled]",
            " [focused]" if self.focused else "",
            " [editable]" if self.editable else "",
        ])
        return f'{self.role.removeprefix("AX")} "{self.label}"{flags}, {self.position}'


@dataclass
class Screen:
    app: str
    window: str
    elements: list[Element]
    capture_ms: float

    def by_id(self, eid: str) -> Element | None:
        return next((e for e in self.elements if e.id == eid), None)


def _attr(el, name):
    try:
        err, val = AXUIElementCopyAttributeValue(el, name, None)
        return val if err == 0 else None
    except Exception:
        return None


def _point(el, name, kind):
    raw = _attr(el, name)
    if raw is None:
        return None
    try:
        ok, val = AXValueGetValue(raw, kind, None)
        return (val.x, val.y) if ok and kind == kAXValueTypeCGPoint else (
            (val.width, val.height) if ok else None
        )
    except Exception:
        return None


def _coarse(rect, bounds) -> str:
    x, y, w, h = rect
    bw, bh = max(bounds[0], 1), max(bounds[1], 1)
    cx, cy = (x + w / 2) / bw, (y + h / 2) / bh
    row = "upper" if cy < 0.33 else "middle" if cy < 0.67 else "lower"
    col = "left" if cx < 0.33 else "center" if cx < 0.67 else "right"
    return f"{row} {col}"


def _label(el) -> str:
    for a in (kAXTitleAttribute, kAXDescriptionAttribute, kAXValueAttribute):
        v = _attr(el, a)
        if isinstance(v, str) and v.strip():
            return v.strip()[:80]
    return ""


def capture(app_name: str | None = None, max_elements: int = 250,
            max_depth: int = 22) -> Screen:
    """Snapshot the frontmost (or named) app's focused window."""
    if not AXIsProcessTrusted():
        raise PermissionError(
            "Accessibility access not granted.\n"
            "  System Settings > Privacy & Security > Accessibility,\n"
            "  then enable the app running this (Terminal, or your IDE).\n"
            "  This is a security setting -- grant it yourself; nothing here changes it."
        )

    ws = NSWorkspace.sharedWorkspace()
    if app_name:
        app = next((a for a in ws.runningApplications()
                    if a.localizedName() == app_name), None)
        if app is None:
            raise LookupError(f"no running app named {app_name!r}")
    else:
        app = ws.frontmostApplication()

    t0 = time.perf_counter()
    root = AXUIElementCreateApplication(app.processIdentifier())
    windows = _attr(root, kAXWindowsAttribute) or []
    if not windows:
        raise LookupError(f"{app.localizedName()} has no accessible windows")
    win = windows[0]
    win_title = _label(win) or "(untitled)"

    origin = _point(win, kAXPositionAttribute, kAXValueTypeCGPoint) or (0, 0)
    size = _point(win, kAXSizeAttribute, kAXValueTypeCGSize) or (1440, 900)

    elements: list[Element] = []
    counter = 0

    def walk(el, depth: int) -> None:
        nonlocal counter
        if depth > max_depth or len(elements) >= max_elements:
            return
        role = _attr(el, kAXRoleAttribute) or ""
        if role in ACTIONABLE:
            label = _label(el)
            if label:
                pos = _point(el, kAXPositionAttribute, kAXValueTypeCGPoint) or origin
                sz = _point(el, kAXSizeAttribute, kAXValueTypeCGSize) or (0, 0)
                rect = (pos[0] - origin[0], pos[1] - origin[1], sz[0], sz[1])
                counter += 1
                elements.append(Element(
                    id=f"e{counter}",
                    label=label,
                    role=role,
                    enabled=bool(_attr(el, kAXEnabledAttribute)),
                    focused=bool(_attr(el, kAXFocusedAttribute)),
                    editable=role in EDITABLE,
                    position=_coarse(rect, size),
                    rect=rect,
                    ref=el,
                ))
        for child in (_attr(el, kAXChildrenAttribute) or []):
            walk(child, depth + 1)

    walk(win, 0)
    return Screen(app.localizedName(), win_title, elements,
                  (time.perf_counter() - t0) * 1000)


# -- execution -------------------------------------------------------------

def press(element: Element) -> bool:
    """Native accessibility press. No synthetic mouse events, no coordinates."""
    return AXUIElementPerformAction(element.ref, kAXPressAction) == 0


def set_text(element: Element, text: str) -> bool:
    """Write the value directly. No keystrokes, no focus change."""
    if not element.editable:
        return False
    return AXUIElementSetAttributeValue(element.ref, kAXValueAttribute, text[:2000]) == 0
