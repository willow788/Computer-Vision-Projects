import cv2
import numpy as np

# Frame width and height (larger webcam window)
w = 1280
h = 720

cap = cv2.VideoCapture(0)
cap.set(3, w)
cap.set(4, h)
cap.set(10, 150)

#hsv ranges for tracking colors
color_ranges = {
    'red': ([0, 120, 70], [10, 255, 255]),
    'green': ([36, 25, 25], [70, 255, 255]),
    'blue': ([94, 80, 2], [126, 255, 255])
}

# More visually pleasing drawing palette (BGR)
draw_colors = [
    [80, 170, 255],   # Orange
    [120, 230, 120],  # Mint
    [255, 140, 90],   # Blue-ish orange tone
    [200, 110, 255],  # Pink
    [90, 220, 220],   # Cyan
    [255, 255, 255]   # White
]

ui_height = 80
brush_radius = 8
points = []  # list of (x, y, color_id)
active_color = 0


def find_tracker_point(img, ranges, out_img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for idx, (lower, upper) in enumerate(ranges):
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # Create mask for the current tracked color range
        mask = cv2.inRange(hsv, lower, upper)
        cx, cy = get_contour_center(mask)

        if cx != 0 and cy != 0:
            # White ring + filled center gives better visibility
            cv2.circle(out_img, (cx, cy), 14, (255, 255, 255), 2)
            cv2.circle(out_img, (cx, cy), 8, draw_colors[idx % len(draw_colors)], cv2.FILLED)
            return cx, cy

    return None

def get_contour_center(mask):

    #get the contours from the mask
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    cx, cy = 0, 0

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, box_w, _ = cv2.boundingRect(approx)
            return x + box_w // 2, y
    return cx, cy


def draw_toolbar(img, selected_color):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, ui_height), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    swatch_size = 55
    gap = 12
    start_x = 20
    y1 = 12
    y2 = y1 + swatch_size

    for i, color in enumerate(draw_colors):
        x1 = start_x + i * (swatch_size + gap)
        x2 = x1 + swatch_size
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
        border = (255, 255, 255) if i == selected_color else (90, 90, 90)
        thickness = 3 if i == selected_color else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), border, thickness)

    clear_x1 = start_x + len(draw_colors) * (swatch_size + gap) + 25
    clear_x2 = clear_x1 + 150
    cv2.rectangle(img, (clear_x1, y1), (clear_x2, y2), (70, 70, 70), cv2.FILLED)
    cv2.rectangle(img, (clear_x1, y1), (clear_x2, y2), (220, 220, 220), 2)
    cv2.putText(img, "CLEAR", (clear_x1 + 25, y1 + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)

    cv2.putText(img, "Move marker to top bar to pick color", (w - 430, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 225, 225), 1)

    return {
        "swatch_size": swatch_size,
        "gap": gap,
        "start_x": start_x,
        "y1": y1,
        "y2": y2,
        "clear_x1": clear_x1,
        "clear_x2": clear_x2
    }


def pick_color_or_action(x, y, toolbar_cfg):
    if y < toolbar_cfg["y1"] or y > toolbar_cfg["y2"]:
        return None

    if toolbar_cfg["clear_x1"] <= x <= toolbar_cfg["clear_x2"]:
        return "clear"

    for i in range(len(draw_colors)):
        x1 = toolbar_cfg["start_x"] + i * (toolbar_cfg["swatch_size"] + toolbar_cfg["gap"])
        x2 = x1 + toolbar_cfg["swatch_size"]
        if x1 <= x <= x2:
            return i

    return None


def draw_on_canvas(img, all_points, bgr_vals):
    for x, y, color_id in all_points:
        cv2.circle(img,
                   (x, y),
                   brush_radius,
                   bgr_vals[color_id],
                   cv2.FILLED)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    result = frame.copy()

    toolbar = draw_toolbar(result, active_color)
    tracked_point = find_tracker_point(frame, color_ranges.values(), result)

    if tracked_point:
        px, py = tracked_point
        selected = pick_color_or_action(px, py, toolbar)
        if selected == "clear":
            points.clear()
        elif isinstance(selected, int):
            active_color = selected
        elif py > ui_height:
            points.append((px, py, active_color))

    if points:
        draw_on_canvas(result, points, draw_colors)

    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

