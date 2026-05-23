import math
import os
import re
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for

from database import add_audit, add_entry, add_exit, add_event, get_db, init_db


try:
    from fast_alpr import ALPR

    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )
    print("✅ GLOBAL ALPR іске қосылды")
except Exception as e:
    alpr = None
    print("❌ ALPR жүктелмеді:", e)

app = Flask(__name__)
app.secret_key = "auezov-parking-secret-key"

try:
    init_db()
    print("✅ Render database initialized")
except Exception as e:
    print("❌ Database init error:", e)

UPLOAD_DIR = "static/captures"
os.makedirs(UPLOAD_DIR, exist_ok=True)

barriers = {
    "cam1": {"fixed_open": False, "fixed_close": False, "state": "closed"},
    "cam2": {"fixed_open": False, "fixed_close": False, "state": "closed"},
}

state = {
    "cam1_plate": "---",
    "cam2_plate": "---",
}

LAPTOP_CAMERA_INDEXES = [0]
DROIDCAM_INDEXES = [1, 2, 3, 4, 5]

DROIDCAM_URLS = [
    "http://10.84.229.42:4747/video",
    "http://10.84.229.42:4747/mjpegfeed",
    "http://10.84.229.42:4747/videofeed",
    "http://10.180.112.227:4747/video",
    "http://10.180.112.227:4747/mjpegfeed",
    "http://10.180.112.227:4747/videofeed",
    "http://10.7.253.217:4747/video",
    "http://10.7.253.217:4747/mjpegfeed",
    "http://10.7.253.217:4747/videofeed",
]

entry_cap = None
exit_cap = None

last_detect_time = {
    "entry": 0,
    "exit": 0,
}

plate_confirm = {
    "entry": {"last": "", "count": 0},
    "exit": {"last": "", "count": 0},
}

def login_required():
    return session.get("logged_in") is True


def normalize_plate(text):
    if not text:
        return ""

    text = str(text).upper()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    text = text.replace("_", "")

    text = re.sub(r"[^A-Z0-9]", "", text)

    bad_words = [
        "PLATE", "PREDICTION", "CHAR", "PROBS", "PROB",
        "REGION", "UNKNOWN", "NONE", "FRANCE",
        "CANADA", "VIETNAM", "LATVIA", "NORWAY",
        "KINGDOM", "DKINGDOM"
    ]

    for word in bad_words:
        text = text.replace(word, "")

    if len(text) < 4 or len(text) > 12:
        return ""

    if text.count("0") >= 7:
        return ""

    has_digit = any(ch.isdigit() for ch in text)
    has_alpha = any(ch.isalpha() for ch in text)

    if not has_digit or not has_alpha:
        return ""

    return text


from datetime import datetime
import math

def calculate_amount_by_time(entry_time, exit_time=None):
    try:
        if not entry_time:
            return 0, 0

        entry_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")

        if exit_time:
            exit_dt = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
        else:
            exit_dt = datetime.now()

        minutes = int((exit_dt - entry_dt).total_seconds() / 60)

        if minutes < 0:
            minutes = 0

        # Алғашқы 2 сағат = 200 тг
        if minutes <= 120:
            amount = 200
        else:
            extra_minutes = minutes - 120
            extra_hours = math.ceil(extra_minutes / 60)
            amount = 200 + (extra_hours * 100)

        return minutes, amount

    except Exception as e:
        print("❌ calculate_amount_by_time қатесі:", e)
        return 0, 0


def encode_frame(frame, remote=False):
    if remote:
        frame = cv2.resize(frame, (640, 360))
        quality = 55
    else:
        frame = cv2.resize(frame, (1280, 720))
        quality = 85

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    if not ok:
        return b""

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" +
        buffer.tobytes() +
        b"\r\n"
    )


def no_signal_frame(text):
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    frame[:] = (10, 15, 30)

    cv2.putText(frame, "Auezov Parking", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, "Camera / DroidCam IP tekseriniz", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)

    return frame


def configure_cap(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def test_capture(cap, attempts=8):
    for _ in range(attempts):
        ok, frame = cap.read()

        if ok and frame is not None and frame.size > 0:
            return True

        time.sleep(0.25)

    return False


def open_laptop_camera():
    for index in LAPTOP_CAMERA_INDEXES:
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                print(f"🔎 Ноутбук камера: index={index}, backend={backend}")
                cap = cv2.VideoCapture(index, backend)
                configure_cap(cap)

                if cap.isOpened() and test_capture(cap):
                    print(f"✅ Ноутбук камера қосылды: index={index}, backend={backend}")
                    return cap

                cap.release()
            except Exception as e:
                print("❌ Ноутбук камера қатесі:", e)

    return None


def open_droidcam():
    for index in DROIDCAM_INDEXES:
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                print(f"🔎 DroidCam index: {index}, backend={backend}")
                cap = cv2.VideoCapture(index, backend)
                configure_cap(cap)

                if cap.isOpened() and test_capture(cap):
                    print(f"✅ DroidCam index арқылы қосылды: {index}")
                    return cap

                cap.release()
            except Exception as e:
                print("❌ DroidCam index қатесі:", e)

    for url in DROIDCAM_URLS:
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
            try:
                print(f"🔎 DroidCam URL: {url}, backend={backend}")
                cap = cv2.VideoCapture(url, backend)
                configure_cap(cap)

                if test_capture(cap):
                    print(f"✅ DroidCam URL арқылы қосылды: {url}")
                    return cap

                cap.release()
            except Exception as e:
                print("❌ DroidCam URL қатесі:", e)

    return None

RENDER_API = "https://auezovparking.xyz/api/sync-entry"


def send_to_render(plate, post, direction, image_path=""):
    try:
        requests.post(
            RENDER_API,
            json={
                "plate": plate,
                "post": post,
                "direction": direction,
                "image_path": image_path
            },
            timeout=5
        )

        print("🌐 Render-ге жіберілді:", plate)

    except Exception as e:
        print("⚠️ Render sync қатесі:", e)


def cleanup_old_captures(limit=100):
    folder = Path("static/captures")

    if not folder.exists():
        return

    files = sorted(
        folder.glob("*.jpg"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    for old_file in files[limit:]:
        try:
            old_file.unlink()
        except:
            pass

def save_frame(frame, camera_name):
    filename = f"{camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, frame)
    cleanup_old_captures(100)
    return path.replace("\\", "/")


def is_blacklisted(plate):
    db = get_db()
    row = db.execute(
        "SELECT * FROM lists WHERE plate = ? AND type = 'black'",
        (plate,),
    ).fetchone()
    db.close()

    return row is not None


def has_free_access(plate):
    today = datetime.now().strftime("%Y-%m-%d")
    db = get_db()

    abonement = db.execute(
        """
        SELECT *
        FROM abonements
        WHERE plate = ?
          AND active = 1
          AND paid = 1
          AND DATE(end_date) >= DATE(?)
        """,
        (plate, today),
    ).fetchone()

    admin = db.execute(
        """
        SELECT *
        FROM administration_access
        WHERE plate = ?
          AND active = 1
        """,
        (plate,),
    ).fetchone()

    db.close()

    if admin:
        return True, "Администрация"

    if abonement:
        return True, "Абонемент"

    return False, ""


def mark_last_log_free(plate, free_type):
    db = get_db()
    db.execute(
        """
        UPDATE vehicle_logs
        SET amount = 0,
            paid = 1,
            note = ?
        WHERE id = (
            SELECT id
            FROM vehicle_logs
            WHERE plate = ?
            ORDER BY id DESC
            LIMIT 1
        )
        """,
        (free_type, plate),
    )
    db.commit()
    db.close()

import re

BAD_OCR_WORDS = (
    "PLATE", "PREDICTION", "CHAR", "PROB", "PROBS", "NONE",
    "REGION", "LICENSE", "TEXT", "SCORE"
)

def normalize_plate(text):
    text = str(text or "").upper()

    replace_map = {
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
        "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
        "У": "Y", "Х": "X",
    }

    for old, new in replace_map.items():
        text = text.replace(old, new)

    text = re.sub(r"[^A-Z0-9]", "", text)

    for bad in BAD_OCR_WORDS:
        text = text.replace(bad, "")

    patterns = [
        r"[A-Z][0-9]{3}[A-Z]{2}[0-9]{2}",    # A222MP77
        r"[A-Z][0-9]{3}[A-Z]{3}",            # A222MPY
        r"[A-Z]{2}[0-9]{4}[A-Z]{2}",         # KA0132CO
        r"[0-9]{3}[A-Z]{3}[0-9]{2}",         # 001DAU13
        r"[A-Z]{1,2}[0-9]{3,4}[A-Z]{1,2}",   # запасной формат
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return ""

def process_plate(frame, camera_name):
    now = time.time()

    if now - last_detect_time[camera_name] < 0.25:
        return state["cam1_plate"] if camera_name == "entry" else state["cam2_plate"]

    last_detect_time[camera_name] = now

    if alpr is None:
        return "---"

    try:
        results = alpr.predict(frame)
        plate = ""

        for item in results:
            try:
                if hasattr(item, "ocr") and item.ocr:
                    raw_text = str(item.ocr.text or "")
                    clean = normalize_plate(raw_text)

                    if clean:
                        plate = clean
                        print("OCR:", raw_text, "=>", plate)
                        break
            except Exception:
                continue

        if len(plate) < 4:
            if camera_name == "entry":
                state["cam1_plate"] = "---"
            else:
                state["cam2_plate"] = "---"
            return "---"

        if plate == plate_confirm[camera_name]["last"]:
            plate_confirm[camera_name]["count"] += 1
        else:
            plate_confirm[camera_name]["last"] = plate
            plate_confirm[camera_name]["count"] = 1

        if plate_confirm[camera_name]["count"] < 2:
            print(f"⏳ Растау: {plate} ({plate_confirm[camera_name]['count']}/2)")
            return state["cam1_plate"] if camera_name == "entry" else state["cam2_plate"]

        if camera_name == "entry":
            if plate == state["cam1_plate"]:
                return plate

            state["cam1_plate"] = plate
            camera_label = "Гл корпус кіріс"
            event_type = "entry"

        else:
            if plate == state["cam2_plate"]:
                return plate

            state["cam2_plate"] = plate
            camera_label = "Гл корпус шығыс"
            event_type = "exit"

        image_path = save_frame(frame, camera_name)
        print("💾 Сақталуда:", plate)

        if is_blacklisted(plate):
            add_event(plate, "Гл корпус", "blacklist", image_path)
            print(f"⛔ BLACKLIST: {plate}")

            plate_confirm[camera_name]["last"] = ""
            plate_confirm[camera_name]["count"] = 0

            return plate

        free, free_type = has_free_access(plate)

        if camera_name == "entry":
            saved = add_entry(plate, camera_label, image_path)

            if saved:
                send_to_render(plate, camera_label, event_type, image_path)

                if free:
                    mark_last_log_free(plate, free_type)

                print(f"✅ ENTRY SAVED TO JOURNAL: {plate}")
            else:
                print(f"⚠️ ENTRY сақталмады, бұл номер already inside болуы мүмкін: {plate}")

        else:
            saved = add_exit(plate, camera_label, image_path)

            if saved:
                send_to_render(plate, camera_label, event_type, image_path)

                if free:
                    mark_last_log_free(plate, free_type)

                print(f"✅ EXIT SAVED TO JOURNAL: {plate}")
            else:
                print(f"⚠️ EXIT сақталмады, журналда inside табылмады: {plate}")

        plate_confirm[camera_name]["last"] = ""
        plate_confirm[camera_name]["count"] = 0

        return plate

    except Exception as e:
        print("❌ ALPR оқу қатесі:", e)
        return state["cam1_plate"] if camera_name == "entry" else state["cam2_plate"]


def draw_overlay(frame, camera_name, plate):
    label = "ENTRY" if camera_name == "entry" else "EXIT"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Auezov Parking | {label} | {plate}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (70, 255, 20),
        2,
    )

    return frame


def get_camera(camera_name):
    global entry_cap, exit_cap

    if camera_name == "entry":
        if entry_cap is None or not entry_cap.isOpened():
            entry_cap = open_laptop_camera()
        return entry_cap

    if exit_cap is None or not exit_cap.isOpened():
        exit_cap = open_droidcam()

    return exit_cap


def reset_camera(camera_name):
    global entry_cap, exit_cap

    if camera_name == "entry":
        if entry_cap:
            entry_cap.release()

        entry_cap = open_laptop_camera()
        return entry_cap

    if exit_cap:
        exit_cap.release()

    exit_cap = open_droidcam()
    return exit_cap


def generate_frames(camera_name, remote=False):
    while True:
        cap = get_camera(camera_name)

        if cap is None:
            text = "Laptop kamera kosylmady" if camera_name == "entry" else "DroidCam kosylmady"
            yield encode_frame(no_signal_frame(text), remote=remote)
            time.sleep(1)
            continue

        ok, frame = cap.read()

        if not ok or frame is None or frame.size == 0:
            reset_camera(camera_name)
            yield encode_frame(no_signal_frame("Kamera kadr bermedi"), remote=remote)
            time.sleep(1)
            continue

        plate = process_plate(frame, camera_name)
        frame = draw_overlay(frame, camera_name, plate)

        yield encode_frame(frame, remote=remote)

        if remote:
            time.sleep(0.2)
        else:
            time.sleep(0.03)


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("login")
        password = request.form.get("password")

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE login = ? AND password = ?",
            (username, password),
        ).fetchone()

        if username == "Auezov" and password == "1943":
            db.execute(
                "INSERT OR REPLACE INTO users(login, password) VALUES (?, ?)",
                ("Auezov", "1943"),
            )
            db.commit()
            user = {"login": "Auezov"}

        db.close()

        if user:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))

        error = "Логин немесе пароль қате"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("dashboard.html")


@app.route("/journal")
def journal():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()

    rows = db.execute("""
        SELECT * FROM vehicle_logs
        ORDER BY id DESC
    """).fetchall()

    updated_logs = []

    for row in rows:
        log = dict(row)

        minutes, amount = calculate_amount_by_time(
            log["entry_time"],
            log["exit_time"]
        )

        log["live_minutes"] = minutes
        log["live_amount"] = amount

        # базаға update
        db.execute("""
            UPDATE vehicle_logs
            SET duration_minutes = ?,
                amount = ?
            WHERE id = ?
        """, (
            minutes,
            amount,
            log["id"]
        ))

        updated_logs.append(log)

    db.commit()
    db.close()

    return render_template(
        "journal.html",
        logs=updated_logs
    )


@app.route("/events")
def events():
    if not login_required():
        return redirect(url_for("login"))

    plate = normalize_plate(request.args.get("plate", ""))
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")

    query = "SELECT * FROM events WHERE 1=1"
    params = []

    if plate:
        query += " AND plate LIKE ?"
        params.append(f"%{plate}%")

    if date_from:
        query += " AND DATE(created_at) >= DATE(?)"
        params.append(date_from)

    if date_to:
        query += " AND DATE(created_at) <= DATE(?)"
        params.append(date_to)

    query += " ORDER BY id DESC"

    db = get_db()
    events_data = db.execute(query, params).fetchall()
    db.close()

    return render_template(
        "events.html",
        events=events_data,
        plate=plate,
        date_from=date_from,
        date_to=date_to,
    )


@app.route("/audit")
def audit():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    logs = db.execute("SELECT * FROM audit_logs ORDER BY id DESC").fetchall()
    db.close()

    return render_template("audit.html", logs=logs)


@app.route("/payments")
def payments():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    payments_data = db.execute("SELECT * FROM payments ORDER BY id DESC").fetchall()
    lists = db.execute("SELECT * FROM lists ORDER BY id DESC").fetchall()
    balances = db.execute("SELECT * FROM balances ORDER BY id DESC").fetchall()
    db.close()

    return render_template(
        "payments.html",
        payments=payments_data,
        lists=lists,
        balances=balances,
    )


@app.route("/abonement")
def abonement():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    rows = db.execute("SELECT * FROM abonements ORDER BY id DESC").fetchall()
    db.close()

    return render_template("abonement.html", abonements=rows)


@app.route("/administration")
def administration():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    rows = db.execute("SELECT * FROM administration_access ORDER BY id DESC").fetchall()
    db.close()

    return render_template("administration.html", staff=rows)


@app.route("/client")
def client():
    return render_template("client.html")



@app.route("/qr")
def qr_page():
    return render_template("qr_links.html")


@app.route("/pay")
def pay_page():
    return render_template("qr_pay.html")


@app.route("/api/qr-check", methods=["POST"])
def qr_check():
    plate = normalize_plate(request.form.get("plate", ""))

    if not plate:
        return jsonify({"ok": False, "message": "Номер енгізіңіз"})

    db = get_db()

    log = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (plate,),
    ).fetchone()

    db.close()

    if not log:
        return jsonify({
            "ok": False,
            "message": "Номер журналда табылмады"
        })

    if int(log["paid"] or 0) == 1:
        return jsonify({
            "ok": True,
            "plate": plate,
            "amount": 0,
            "minutes": log["duration_minutes"],
            "message": "Төлем жасалған"
        })

    minutes, amount = calculate_amount_by_time(
        log["entry_time"],
        log["exit_time"],
    )

    return jsonify({
        "ok": True,
        "plate": plate,
        "amount": amount,
        "minutes": minutes,
    })


@app.route("/api/qr-pay", methods=["POST"])
def qr_pay():
    plate = normalize_plate(request.form.get("plate", ""))
    amount = int(request.form.get("amount", 0))

    if not plate:
        return jsonify({
            "ok": False,
            "message": "Номер жоқ"
        })

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db = get_db()

    db.execute(
        """
        INSERT INTO payments(
            plate,
            amount,
            method,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            plate,
            amount,
            "QR DEMO",
            now,
        ),
    )

    db.execute(
        """
        UPDATE vehicle_logs
        SET paid = 1
        WHERE id = (
            SELECT id
            FROM vehicle_logs
            WHERE plate = ?
            ORDER BY id DESC
            LIMIT 1
        )
        """,
        (plate,),
    )

    db.commit()
    db.close()

    add_audit(
        plate,
        "QR төлем",
        "",
        f"{amount} ₸",
        "Demo payment"
    )

    return jsonify({
        "ok": True
    })


@app.route("/video/<camera>")
def video(camera):
    if camera not in ["entry", "exit"]:
        return "Camera not found", 404

    return Response(
        generate_frames(camera, remote=False),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/remote-video/<camera>")
def remote_video(camera):
    if camera not in ["entry", "exit"]:
        return "Camera not found", 404

    return Response(
        generate_frames(camera, remote=True),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/last-plates")
def api_last_plates():
    return jsonify({
        "entry": state["cam1_plate"],
        "exit": state["cam2_plate"],
    })


@app.route("/api/manual-entry", methods=["POST"])
def manual_entry():
    plate = normalize_plate(request.form.get("plate", ""))
    direction = request.form.get("direction")

    if not plate:
        return jsonify({"ok": False, "message": "Номер жазыңыз"})

    free, free_type = has_free_access(plate)

    if direction == "entry":
        add_entry(plate, "Қолмен кіргізілді", None)

        if free:
            mark_last_log_free(plate, free_type)

        add_audit(plate, "Manual entry", "", plate, f"Қолмен кіргізілді | {free_type if free else 'Ақылы'}")
        state["cam1_plate"] = plate

    elif direction == "exit":
        success = add_exit(plate, "Қолмен шығарылды", None)

        if success and free:
            mark_last_log_free(plate, free_type)

        add_audit(plate, "Manual exit", "", plate, f"Қолмен шығарылды | {free_type if free else 'Ақылы'}")
        state["cam2_plate"] = plate

        if not success:
            return jsonify({"ok": False, "message": "Бұл номер ішкі журналда табылмады"})
    else:
        return jsonify({"ok": False, "message": "Бағыт қате"})

    return jsonify({"ok": True})


@app.route("/api/barrier", methods=["POST"])
def barrier():
    name = request.form.get("name")
    action = request.form.get("action")
    reason = request.form.get("reason", "")

    if name not in ["entry", "exit"]:
        return jsonify({"ok": False, "message": "Шлагбаум аты қате"})

    key = "cam1" if name == "entry" else "cam2"

    if action == "open":
        barriers[key]["state"] = "open"
    elif action == "close":
        barriers[key]["state"] = "closed"
    elif action == "fix_open":
        barriers[key]["state"] = "open"
        barriers[key]["fixed_open"] = True
        barriers[key]["fixed_close"] = False
    elif action == "fix_close":
        barriers[key]["state"] = "closed"
        barriers[key]["fixed_close"] = True
        barriers[key]["fixed_open"] = False
    elif action == "unfix":
        barriers[key]["fixed_open"] = False
        barriers[key]["fixed_close"] = False
    else:
        return jsonify({"ok": False, "message": "Action қате"})

    fixed_state = "none"

    if barriers[key]["fixed_open"]:
        fixed_state = "open"

    if barriers[key]["fixed_close"]:
        fixed_state = "closed"

    db = get_db()
    db.execute(
        "UPDATE barriers SET state = ?, fixed_state = ? WHERE name = ?",
        (barriers[key]["state"], fixed_state, name),
    )
    db.commit()
    db.close()

    add_audit(name, f"Barrier {action}", "", barriers[key]["state"], reason)

    return jsonify({"ok": True})


@app.route("/api/abonement", methods=["POST"])
def api_abonement():
    plate = normalize_plate(request.form.get("plate", ""))
    full_name = request.form.get("full_name", "").strip()
    phone = request.form.get("phone", "").strip()
    group_name = request.form.get("group_name", "").strip()

    if not plate or not full_name or not phone:
        return jsonify({"ok": False, "message": "Аты-жөні, телефон, номер міндетті"})

    now = datetime.now()
    start_date = now.strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")

    db = get_db()
    db.execute(
        """
        INSERT OR REPLACE INTO abonements(
            full_name,
            phone,
            plate,
            group_name,
            price,
            start_date,
            end_date,
            active,
            created_at
        )
        VALUES (?, ?, ?, ?, 5000, ?, ?, 1, ?)
        """,
        (full_name, phone, plate, group_name, start_date, end_date, created_at),
    )
    db.commit()
    db.close()

    add_audit(plate, "Абонемент қосылды", "", f"{full_name} | 5000 ₸ | {end_date}", "1 айлық абонемент")

    return jsonify({"ok": True})

@app.route("/api/abonement/pay", methods=["POST"])
def pay_abonement():
    plate = normalize_plate(request.form.get("plate", ""))

    if not plate:
        return jsonify({"ok": False, "message": "Номер жоқ"})

    db = get_db()

    row = db.execute(
        "SELECT * FROM abonements WHERE plate = ?",
        (plate,),
    ).fetchone()

    if not row:
        db.close()
        return jsonify({"ok": False, "message": "Абонемент табылмады"})

    db.execute(
        """
        UPDATE abonements
        SET paid = 1,
            active = 1
        WHERE plate = ?
        """,
        (plate,),
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.execute(
        """
        INSERT INTO payments(plate, amount, method, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (plate, 5000, "Абонемент", now),
    )

    db.commit()
    db.close()

    add_audit(
        plate,
        "Абонемент төлемі",
        "Төленбеді",
        "Төленді",
        "5000 ₸ абонемент төленді",
    )

    return jsonify({"ok": True})

@app.route("/api/administration", methods=["POST"])
def api_administration():
    plate = normalize_plate(request.form.get("plate", ""))
    full_name = request.form.get("full_name", "").strip()
    phone = request.form.get("phone", "").strip()
    position = request.form.get("position", "").strip()

    if not plate or not full_name or not phone or not position:
        return jsonify({"ok": False, "message": "Барлық мәліметті толтырыңыз"})

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db = get_db()
    db.execute(
        """
        INSERT OR REPLACE INTO administration_access(
            full_name,
            phone,
            plate,
            position,
            active,
            created_at
        )
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (full_name, phone, plate, position, now),
    )
    db.commit()
    db.close()

    add_audit(plate, "Администрация access", "", f"{full_name} | {position}", "Тегін кіру рұқсаты")

    return jsonify({"ok": True})


@app.route("/api/free-access/delete", methods=["POST"])
def delete_free_access():
    table = request.form.get("table")
    plate = normalize_plate(request.form.get("plate", ""))

    if table not in ["abonements", "administration_access"]:
        return jsonify({"ok": False})

    db = get_db()
    db.execute(f"UPDATE {table} SET active = 0 WHERE plate = ?", (plate,))
    db.commit()
    db.close()

    add_audit(plate, "Free access disabled", table, "active=0", "Рұқсат өшірілді")

    return jsonify({"ok": True})


@app.route("/api/journal/edit-plate", methods=["POST"])
def edit_plate():
    old_plate = normalize_plate(request.form.get("old_plate", ""))
    new_plate = normalize_plate(request.form.get("new_plate", ""))

    if not old_plate or not new_plate:
        return jsonify({"ok": False, "message": "Номер дұрыс емес"})

    db = get_db()
    db.execute("UPDATE vehicle_logs SET plate = ? WHERE plate = ?", (new_plate, old_plate))
    db.execute("UPDATE events SET plate = ? WHERE plate = ?", (new_plate, old_plate))
    db.execute("UPDATE payments SET plate = ? WHERE plate = ?", (new_plate, old_plate))
    db.execute("UPDATE balances SET plate = ? WHERE plate = ?", (new_plate, old_plate))
    db.execute("UPDATE abonements SET plate = ? WHERE plate = ?", (new_plate, old_plate))
    db.execute("UPDATE administration_access SET plate = ? WHERE plate = ?", (new_plate, old_plate))
    db.commit()
    db.close()

    add_audit(new_plate, "Изменить номер", old_plate, new_plate, f"{old_plate} → {new_plate}")

    return jsonify({"ok": True})


@app.route("/api/journal/edit-time", methods=["POST"])
def edit_time():
    log_id = request.form.get("log_id")
    entry_time = request.form.get("entry_time")
    exit_time = request.form.get("exit_time")

    if not exit_time:
        exit_time = None

    db = get_db()
    old = db.execute("SELECT * FROM vehicle_logs WHERE id = ?", (log_id,)).fetchone()

    if not old:
        db.close()
        return jsonify({"ok": False, "message": "Журнал табылмады"})

    minutes, amount = calculate_amount_by_time(entry_time, exit_time)
    status = "exited" if exit_time else "inside"

    free, free_type = has_free_access(old["plate"])

    if free:
        amount = 0
        paid = 1
        note = free_type
    else:
        paid = old["paid"]
        note = old["note"]

    db.execute(
        """
        UPDATE vehicle_logs
        SET entry_time = ?,
            exit_time = ?,
            duration_minutes = ?,
            amount = ?,
            status = ?,
            paid = ?,
            note = ?
        WHERE id = ?
        """,
        (entry_time, exit_time, minutes, amount, status, paid, note, log_id),
    )

    db.commit()
    db.close()

    add_audit(
        old["plate"],
        "Изменить время",
        f"{old['entry_time']} / {old['exit_time']}",
        f"{entry_time} / {exit_time} | {amount} ₸",
        "Кіру/шығу уақыты және сома қайта есептелді",
    )

    return jsonify({"ok": True})


@app.route("/api/journal/delete-debt", methods=["POST"])
def delete_debt():
    log_id = request.form.get("log_id")

    db = get_db()
    row = db.execute("SELECT * FROM vehicle_logs WHERE id = ?", (log_id,)).fetchone()

    if not row:
        db.close()
        return jsonify({"ok": False, "message": "Журнал табылмады"})

    db.execute("UPDATE vehicle_logs SET amount = 0, paid = 1 WHERE id = ?", (log_id,))
    db.commit()
    db.close()

    add_audit(row["plate"], "Удалить задолженность", str(row["amount"]), "0", "Қарыз өшірілді")

    return jsonify({"ok": True})


@app.route("/api/journal/delete-log", methods=["POST"])
def delete_log():
    log_id = request.form.get("log_id")

    db = get_db()
    row = db.execute("SELECT * FROM vehicle_logs WHERE id = ?", (log_id,)).fetchone()

    if not row:
        db.close()
        return jsonify({"ok": False, "message": "Журнал табылмады"})

    db.execute("DELETE FROM vehicle_logs WHERE id = ?", (log_id,))
    db.commit()
    db.close()

    add_audit(row["plate"], "Удалить из журнала", str(dict(row)), "", "Журналдан өшірілді")

    return jsonify({"ok": True})


@app.route("/api/list", methods=["POST"])
def api_list():
    plate = normalize_plate(request.form.get("plate", ""))
    list_type = request.form.get("type")
    reason = request.form.get("reason", "")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not plate or list_type not in ["white", "black"]:
        return jsonify({"ok": False})

    db = get_db()
    db.execute(
        """
        INSERT OR REPLACE INTO lists(plate, type, reason, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (plate, list_type, reason, now),
    )
    db.commit()
    db.close()

    add_audit(plate, f"Add {list_type} list", "", list_type, reason)

    return jsonify({"ok": True})


@app.route("/api/payment", methods=["POST"])
def api_payment():
    plate = normalize_plate(request.form.get("plate", ""))
    amount = int(request.form.get("amount", 0))
    method = request.form.get("method", "Kaspi")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db = get_db()
    db.execute(
        "INSERT INTO payments(plate, amount, method, created_at) VALUES (?, ?, ?, ?)",
        (plate, amount, method, now),
    )

    db.execute(
        """
        UPDATE vehicle_logs
        SET paid = 1
        WHERE id = (
            SELECT id
            FROM vehicle_logs
            WHERE plate = ?
            ORDER BY id DESC
            LIMIT 1
        )
        """,
        (plate,),
    )

    db.commit()
    db.close()

    add_audit(plate, "Payment", "", f"{amount} ₸", method)

    return jsonify({"ok": True})


@app.route("/api/balance", methods=["POST"])
def api_balance():
    plate = normalize_plate(request.form.get("plate", ""))
    amount = int(request.form.get("amount", 0))
    operation = request.form.get("operation")

    if operation == "minus":
        amount = -amount

    db = get_db()
    db.execute("INSERT OR IGNORE INTO balances(plate, balance) VALUES (?, 0)", (plate,))
    db.execute("UPDATE balances SET balance = balance + ? WHERE plate = ?", (amount, plate))
    db.commit()
    db.close()

    add_audit(plate, "Balance", "", str(amount), operation)

    return jsonify({"ok": True})

@app.route("/api/sync-entry", methods=["POST"])
def api_sync_entry():
    data = request.get_json(silent=True) or {}

    plate = normalize_plate(data.get("plate", ""))
    post = data.get("post", "Гл корпус кіріс")
    direction = data.get("direction", "entry")
    image_path = data.get("image_path", "")

    if not plate:
        return {"ok": False, "error": "plate required"}, 400

    if direction == "exit":
        ok = add_exit(plate, post, image_path)
        return {"ok": True, "plate": plate, "direction": "exit", "saved": ok}

    add_entry(plate, post, image_path)
    return {"ok": True, "plate": plate, "direction": "entry", "saved": True}


@app.route("/api/client-status", methods=["POST"])
def client_status():
    plate = normalize_plate(request.form.get("plate", ""))
    messages = []

    db = get_db()

    log = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (plate,),
    ).fetchone()

    listed = db.execute("SELECT * FROM lists WHERE plate = ?", (plate,)).fetchone()
    balance = db.execute("SELECT * FROM balances WHERE plate = ?", (plate,)).fetchone()
    db.close()

    if not plate:
        messages.append("Номер енгізілмеді.")
    else:
        free, free_type = has_free_access(plate)

        if free:
            messages.append(f"Сізде тегін кіру рұқсаты бар: {free_type}.")
            messages.append("Қарыз жоқ.")

        elif not log:
            messages.append("Сіздің номеріңіз журналда табылмады.")

        else:
            minutes, real_amount = calculate_amount_by_time(log["entry_time"], log["exit_time"])
            paid = int(log["paid"] or 0)

            if paid == 0:
                messages.append(f"Төлеу керек сома: {real_amount} ₸")
                messages.append(f"Парковкада тұрған уақыт: {minutes} минут")
            else:
                messages.append("Төлем жасалған. Қарыз жоқ.")

    if listed:
        if listed["type"] == "black":
            messages.append("Сіздің номеріңіз қара тізімде тұр.")
        elif listed["type"] == "white":
            messages.append("Сіздің номеріңіз ақ тізімде. Тегін өте аласыз.")

    if balance:
        messages.append(f"Сіздің баланс: {balance['balance']} ₸")

    if not messages:
        messages.append("Мәселе табылған жоқ. Барлығы дұрыс.")

    return jsonify({
        "ok": True,
        "plate": plate,
        "messages": messages,
    })

@app.route("/api/remote-entry", methods=["POST"])
def remote_entry():
    secret = request.form.get("secret", "")

    if secret != "auezov-secret-2026":
        return jsonify({"ok": False, "message": "Құпия кілт қате"}), 403

    plate = normalize_plate(request.form.get("plate", ""))
    direction = request.form.get("direction", "entry")

    if not plate:
        return jsonify({"ok": False, "message": "Номер жоқ"})

    if direction == "entry":
        add_entry(plate, "Remote camera", None)
    elif direction == "exit":
        add_exit(plate, "Remote camera", None)
    else:
        return jsonify({"ok": False, "message": "direction қате"})

    return jsonify({"ok": True, "plate": plate})

if __name__ == "__main__":
    init_db()
    entry_cap = open_laptop_camera()
    exit_cap = open_droidcam()

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000,
        threaded=True,
        use_reloader=False,
    )