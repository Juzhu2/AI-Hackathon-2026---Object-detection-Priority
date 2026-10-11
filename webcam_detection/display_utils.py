"""Compose annotated video above a separate, wrapped detection text panel."""

import textwrap


def compose_display(cv2, image, record, profile, config, caption=""):
    import numpy as np

    width = image.shape[1]
    scale = min(0.6, max(0.3, width / 1600))
    columns = max(12, int((width - 24) / (20 * scale)))
    lines = [
        f"FPS: {record['fps']:.1f} | tracked: {record['tracked_count']} | {profile}",
        f"Buffer: {config.get('track_buffer', '-')} | ReID: {config.get('with_reid', False)}",
    ]
    for item in record["objects"]:
        label = item.get("object_id") or item["class"]
        line = f"{label} | confidence {item['confidence']:.0%}"
        if item.get("depth"):
            depth = item["depth"]
            line += f" | distance {depth['value']:.2f} {depth['unit']}"
        lines.append(line)
    if not record["objects"]:
        lines.append("No objects detected.")
    if caption:
        lines.append(f"Speech: {caption}")
    wrapped = [part for line in lines for part in textwrap.wrap(line, columns)]
    spacing = max(20, int(36 * scale))
    panel = np.full((24 + spacing * len(wrapped), width, 3), 24, dtype=image.dtype)
    for index, line in enumerate(wrapped):
        cv2.putText(panel, line, (12, 20 + index * spacing), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack((image, panel))
