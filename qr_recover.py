#!/usr/bin/env python3
"""Robust QR recovery cascade: decoder -> ASCII-label OCR -> deskew+retry.

recover(gray) -> (value, method) where method in
  qr:cv2 | qr:zbar | qr:crop | ascii | deskew:qr | deskew:ascii | None
"""
import re
import cv2, numpy as np
from pyzbar.pyzbar import decode as _zbar
import pytesseract

_DET = cv2.QRCodeDetector()
PAT = re.compile(r"HRW-[A-Z]-V\d+-[PX]\d+(?:-S[0-9X]+)?-[FB]")
_TCFG = "--psm 7 -c tessedit_char_whitelist=HRWAVPSXFB0123456789-"


def _rots(g):
    return [np.ascontiguousarray(np.rot90(g, k)) for k in range(4)]


def _decode_multi(g):
    for r in _rots(g):
        v = _DET.detectAndDecode(r)[0]
        if v:
            return v, "qr:cv2"
        z = _zbar(r)
        if z:
            return z[0].data.decode("utf-8", "replace"), "qr:zbar"
    return None, None


def _qr_bbox(g):
    """Return (rotated_gray, bbox) at the rotation where detect() localizes."""
    for r in _rots(g):
        ok, pts = _DET.detect(r)
        if ok and pts is not None:
            p = pts.reshape(-1, 2)
            return r, (int(p[:, 0].min()), int(p[:, 1].min()),
                       int(p[:, 0].max()), int(p[:, 1].max()))
    return None, None


def _decode_cropped(g):
    r, bb = _qr_bbox(g)
    if bb is None:
        return None, None
    x0, y0, x1, y1 = bb
    pad = int(0.5 * max(x1 - x0, y1 - y0, 1))
    crop = r[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    if crop.size == 0:
        return None, None
    up = cv2.resize(crop, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    v, m = _decode_multi(up)
    return (v, "qr:crop") if v else (None, None)


def _ascii_label(g):
    """OCR the printed ASCII label. Try beside a localized QR, else scan the
    bottom band (and 180-flip)."""
    cands = []
    r, bb = _qr_bbox(g)
    if bb is not None:
        x0, y0, x1, y1 = bb
        hh = y1 - y0
        strip = r[max(0, y0 - hh // 3): y1 + hh // 3, x1: min(r.shape[1], x1 + 6 * (x1 - x0))]
        if strip.size:
            cands.append(strip)
    # fallback: bottom 12% band, upright and flipped
    for base in (g, np.ascontiguousarray(np.rot90(g, 2))):
        h, w = base.shape
        cands.append(base[int(0.88 * h):h, 0: int(0.55 * w)])
    for c in cands:
        up = cv2.resize(c, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        for img in (up, cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]):
            txt = pytesseract.image_to_string(img, config=_TCFG)
            m = PAT.search(txt.replace(" ", "").replace("\n", ""))
            if m:
                return m.group(0), "ascii"
    return None, None


def _deskew(g):
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200,
                            minLineLength=g.shape[1] // 3, maxLineGap=20)
    if lines is None:
        return None
    angs = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines[:, 0]]
    angs = [a for a in angs if abs(a) < 20]
    if not angs:
        return None
    a = float(np.median(angs))
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def recover(gray):
    # Stage 1: decoder (full page, then localize+crop+upscale)
    v, m = _decode_multi(gray)
    if v:
        return v, m
    v, m = _decode_cropped(gray)
    if v:
        return v, m
    # Stage 2: ASCII label OCR
    v, m = _ascii_label(gray)
    if v:
        return v, m
    # Stage 3: deskew, then retry decoder + ASCII
    d = _deskew(gray)
    if d is not None:
        v, m = _decode_multi(d)
        if v:
            return v, "deskew:qr"
        v, m = _decode_cropped(d)
        if v:
            return v, "deskew:qr"
        v, m = _ascii_label(d)
        if v:
            return v, "deskew:ascii"
    return None, None
