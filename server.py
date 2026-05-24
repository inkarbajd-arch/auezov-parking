import os
import re
import time
import math
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    print("⚠️ python-dotenv орнатылмаған, Render ENV қолданылады.")

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from database import (
    add_audit,
    add_entry,
    add_exit,
    add_event,
    get_db,
    init_db,
)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
IS_RENDER = os.getenv("RENDER", "").lower() == "true"

# Render-де камера ашылмайды. Камера тек локал компьютерде жұмыс істейді.
# Егер керек болса .env ішіне ENABLE_CAMERA=1 деп қоясың.
ENABLE_CAMERA = os.getenv("ENABLE_CAMERA", "1").strip() == "1"
ENABLE_RENDER_SYNC = os.getenv("ENABLE_RENDER_SYNC", "0").strip() == "1"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "captures"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "auezov-parking-secret-key")


# ============================================================
# DATABASE INIT
# ============================================================

try:
    init_db()
    print("✅ Database initialized")
    print("✅ DB mode:", "PostgreSQL" if DATABASE_URL else "SQLite fallback")
except Exception as e:
    print("❌ Database init error:", e)


# ============================================================
# ALPR INIT
# ============================================================

alpr = None

if ENABLE_CAMERA and not IS_RENDER:
    try:
        from fast_alpr import ALPR

        alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            detector_conf_thresh=0.15,
            ocr_model="cct-xs-v2-global-model",
            ocr_device="cpu",
        )
        print("✅ GLOBAL ALPR іске қосылды")
    except Exception as e:
        alpr = None
        print("❌ ALPR жүктелмеді:", e)
else:
    print("ℹ️ Render немесе камера өшірулі: ALPR жүктелмейді")


# ============================================================
# GLOBAL STATE
# ============================================================

barriers = {
    "entry": {"fixed_open": False, "fixed_close": False, "state": "closed"},
    "exit": {"fixed_open": False, "fixed_close": False, "state": "closed"},
}

state = {
    "entry_plate": "---",
    "exit_plate": "---",
    "entry_frame": None,
    "exit_frame": None,
    "entry_connected": False,
    "exit_connected": False,
    "entry_message": "Камера күтілуде",
    "exit_message": "Камера күтілуде",
}

state_lock = threading.Lock()

last_saved_plate = {
    "entry": "",
    "exit": "",
}

last_saved_time = {
    "entry": 0,
    "exit": 0,
}

plate_confirm = {
    "entry": {"last": "", "count": 0},
    "exit": {"last": "", "count": 0},
}

workers_started = {
    "entry": False,
    "exit": False,
}

worker_threads = {
    "entry": None,
    "exit": None,
}

alpr_busy = {
    "entry": False,
    "exit": False,
}

# ============================================================
# CAMERA SETTINGS
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def login_required():
    return session.get("logged_in") is True


KZ_TZ = timezone(timedelta(hours=5))

def now_str():
    return datetime.now(KZ_TZ).strftime("%Y-%m-%d %H:%M:%S")


BAD_OCR_WORDS = (
    "PLATE",
    "PREDICTION",
    "CHAR",
    "PROB",
    "PROBS",
    "NONE",
    "REGION",
    "LICENSE",
    "TEXT",
    "SCORE",
    "CANADA",
    "VIETNAM",
    "KINGDOM",
    "LATVIA",
    "NORWAY",
    "AUEZOV",
    "PARKING",
)


def normalize_plate(text):
    """
    OCR-дан келген мәтінді номер форматына келтіреді.
    Кирилл әріптерін латынға ауыстырады.
    """
    text = str(text or "").upper()

    replace_map = {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }

    for old, new in replace_map.items():
        text = text.replace(old, new)

    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    text = text.replace(".", "")

    text = re.sub(r"[^A-Z0-9]", "", text)

    for bad in BAD_OCR_WORDS:
        text = text.replace(bad, "")

    patterns = [
        r"\d{3}[A-Z]{3}\d{2}",            # Қазақстан: 858SKA02
        r"\d{3}[A-Z]{2,3}\d{2}",          # Қазақстан OCR вариация
        r"[A-Z]\d{3}[A-Z]{2}\d{2,3}",     # Ресей: A222MP77
        r"[A-Z][0-9]{3}[A-Z]{3}",         # A222MPY
        r"[A-Z]{2}\d{4}[A-Z]{2}",         # KA0132CO
        r"[A-Z]{1,3}\d{3,5}",             # BT4732
        r"\d{3}[A-Z]\d{3,4}",             # 147D0647
        r"[A-Z]{1,3}\d{2,4}[A-Z]{1,3}",   # mixed
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return ""


def is_good_plate(plate):
    plate = normalize_plate(plate)

    if len(plate) < 5 or len(plate) > 10:
        return False

    if any(word in plate for word in BAD_OCR_WORDS):
        return False

    if not any(ch.isdigit() for ch in plate):
        return False

    if not any(ch.isalpha() for ch in plate):
        return False

    return True


def calculate_amount_by_time(entry_time, exit_time=None):
    try:
        if not entry_time:
            return 0, 0

        entry_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")

        if exit_time:
            exit_dt = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
        else:
            exit_dt = datetime.now(KZ_TZ).replace(tzinfo=None)

        minutes = max(int((exit_dt - entry_dt).total_seconds() // 60), 0)

        if minutes <= 120:
            amount = 200
        else:
            extra_minutes = minutes - 120
            extra_hours = math.ceil(extra_minutes / 60)
            amount = 200 + extra_hours * 100

        return minutes, amount

    except Exception as e:
        print("❌ calculate_amount_by_time қатесі:", e)
        return 0, 0


def cleanup_old_captures(limit=200):
    try:
        files = sorted(
            UPLOAD_DIR.glob("*.jpg"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        for old_file in files[limit:]:
            try:
                old_file.unlink()
            except Exception:
                pass

    except Exception as e:
        print("⚠️ cleanup_old_captures қатесі:", e)


def save_frame(frame, camera_name):
    filename = f"{camera_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = UPLOAD_DIR / filename

    cv2.imwrite(str(path), frame)
    cleanup_old_captures(200)

    return f"/static/captures/{filename}"


def encode_frame(frame, remote=False):
    try:
        if remote:
            frame = cv2.resize(frame, (640, 360))
            quality = 55
        else:
            frame = cv2.resize(frame, (854, 480))
            quality = 65

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )

        if not ok:
            return b""

        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    except Exception as e:
        print("❌ encode_frame қатесі:", e)
        return b""


def no_signal_frame(text):
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    frame[:] = (10, 15, 30)

    cv2.putText(
        frame,
        "Auezov Parking",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
    )

    cv2.putText(
        frame,
        text,
        (40, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Camera / DroidCam IP tekseriniz",
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 220, 255),
        2,
    )

    return frame


def draw_overlay(frame, camera_name, plate):
    label = "ENTRY" if camera_name == "entry" else "EXIT"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 62), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Auezov Parking | {label} | {plate}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (70, 255, 20),
        2,
    )

    return frame


def send_to_render(plate, camera, event_type, image_path=None):
    """
    Егер локал мен домен бір PostgreSQL базаға қосылып тұрса,
    Render sync қажет емес. Сондықтан әдепкіде өшірулі.

    Қосу керек болса .env ішінде:
    ENABLE_RENDER_SYNC=1
    """
    enable_sync = os.getenv("ENABLE_RENDER_SYNC", "0").strip() == "1"

    if not enable_sync:
        return False

    try:
        render_url = os.getenv(
            "RENDER_REMOTE_URL",
            "https://auezovparking.xyz/api/remote-entry",
        )

        data = {
            "secret": os.getenv("REMOTE_SECRET", "auezov-secret-2026"),
            "plate": plate,
            "direction": event_type,
            "camera": camera,
        }

        files = None
        real_image_path = None

        if image_path:
            if image_path.startswith("/static/"):
                real_image_path = BASE_DIR / image_path.lstrip("/")
            else:
                real_image_path = Path(image_path)

        if real_image_path and real_image_path.exists():
            files = {"image": open(real_image_path, "rb")}

        response = requests.post(
            render_url,
            data=data,
            files=files,
            timeout=4,
        )

        if files:
            files["image"].close()

        print("☁️ Render sync:", response.status_code, response.text[:120])

        return response.status_code == 200

    except Exception as e:
        print("⚠️ Render sync қатесі:", e)
        return False


def upload_capture_to_render(plate, camera_name, image_path):
    """
    Локалда сақталған суретті Render серверге жібереді.
    Render суретті static/captures ішіне сақтап,
    PostgreSQL-дегі соңғы event және vehicle_logs image_path мәнін жаңартады.
    """
    try:
        if not image_path:
            return False

        real_image_path = None

        if image_path.startswith("/static/"):
            real_image_path = BASE_DIR / image_path.lstrip("/")
        else:
            real_image_path = Path(image_path)

        if not real_image_path.exists():
            print("⚠️ Upload image табылмады:", real_image_path)
            return False

        upload_url = os.getenv(
            "RENDER_UPLOAD_URL",
            "https://auezovparking.xyz/api/upload-capture",
        )

        data = {
            "secret": os.getenv("REMOTE_SECRET", "auezov-secret-2026"),
            "plate": plate,
            "camera_name": camera_name,
        }

        with open(real_image_path, "rb") as f:
            files = {"image": f}

            response = requests.post(
                upload_url,
                data=data,
                files=files,
                timeout=6,
            )

        print("☁️ Capture upload:", response.status_code, response.text[:120])

        return response.status_code == 200

    except Exception as e:
        print("⚠️ Capture upload қатесі:", e)
        return False


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

    white_list = db.execute(
        """
        SELECT *
        FROM lists
        WHERE plate = ?
          AND type = 'white'
        """,
        (plate,),
    ).fetchone()

    db.close()

    if admin:
        return True, "Администрация"

    if abonement:
        return True, "Абонемент"

    if white_list:
        return True, "Ақ список"

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

    # ============================================================
# CAMERA OPEN FUNCTIONS
# ============================================================

def configure_cap(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def test_capture(cap, attempts=8):
    for _ in range(attempts):
        ok, frame = cap.read()

        if ok and frame is not None and frame.size > 0:
            return True

        time.sleep(0.2)

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
    # 1) Алдымен DroidCam virtual camera index арқылы іздейміз
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

    # 2) Егер index арқылы табылмаса, URL арқылы іздейміз
    for url in DROIDCAM_URLS:
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
            try:
                print(f"🔎 DroidCam URL: {url}, backend={backend}")

                cap = cv2.VideoCapture(url, backend)
                configure_cap(cap)

                if cap.isOpened() and test_capture(cap):
                    print(f"✅ DroidCam URL арқылы қосылды: {url}")
                    return cap

                cap.release()

            except Exception as e:
                print("❌ DroidCam URL қатесі:", e)

    return None


def open_camera(camera_name):
    if not ENABLE_CAMERA or IS_RENDER:
        return None

    if camera_name == "entry":
        return open_laptop_camera()

    if camera_name == "exit":
        return open_droidcam()

    return None


# ============================================================
# ALPR PROCESSING
# ============================================================

def read_plate_from_frame(frame):
    """
    Тек соңғы кадрдан номер оқиды.
    Бұл функция video stream ішінде емес, background worker ішінде шақырылады.
    """
    if alpr is None or frame is None:
        return ""

    try:
        results = alpr.predict(frame)
        best_plate = ""

        for item in results:
            if not hasattr(item, "ocr") or not item.ocr:
                continue

            raw_text = str(item.ocr.text or "")
            clean = normalize_plate(raw_text)

            if is_good_plate(clean):
                best_plate = clean
                print("✅ OCR GOOD:", raw_text, "=>", best_plate)
                break

            print("⚠️ OCR REJECT:", raw_text, "=>", clean)

        return best_plate

    except Exception as e:
        print("❌ ALPR оқу қатесі:", e)
        return ""


def save_detected_plate(plate, camera_name, frame):
    """
    Расталған номерді PostgreSQL базаға сақтайды.
    Локалда сақталған суретті Render серверге upload қылады,
    сонда доменде /events бетінде сурет ашылады.
    """
    if not is_good_plate(plate):
        return

    now = time.time()

    # Бір номерді қайта-қайта сақтап тастамау үшін cooldown
    if (
        last_saved_plate[camera_name] == plate
        and now - last_saved_time[camera_name] < 12
    ):
        return

    last_saved_plate[camera_name] = plate
    last_saved_time[camera_name] = now

    if camera_name == "entry":
        camera_label = "Гл корпус кіріс"
        event_type = "entry"
    else:
        camera_label = "Гл корпус шығыс"
        event_type = "exit"

    image_path = save_frame(frame, camera_name)
    print("💾 Сақталуда:", plate, camera_name)

    # 1) BLACKLIST
    if is_blacklisted(plate):
        add_event(plate, "Гл корпус", "blacklist", image_path)

        # Event базаға жазылғаннан кейін суретті доменге upload қыламыз
        upload_capture_to_render(plate, camera_name, image_path)

        print(f"⛔ BLACKLIST: {plate}")
        return

    free, free_type = has_free_access(plate)

    # 2) ENTRY
    if camera_name == "entry":
        saved = add_entry(plate, camera_label, image_path)

        if saved:
            if free:
                mark_last_log_free(plate, free_type)

            # add_entry event жазғаннан кейін ғана upload
            upload_capture_to_render(plate, camera_name, image_path)

            print(f"✅ ENTRY SAVED: {plate}")

        else:
            print(f"⚠️ ENTRY сақталмады: {plate} already inside болуы мүмкін")

    # 3) EXIT
    else:
        saved = add_exit(plate, camera_label, image_path)

        if saved:
            if free:
                mark_last_log_free(plate, free_type)

            # add_exit event жазғаннан кейін ғана upload
            upload_capture_to_render(plate, camera_name, image_path)

            print(f"✅ EXIT SAVED: {plate}")

        else:
            add_event(plate, camera_label, "exit_not_found", image_path)

            # exit_not_found event жазылғаннан кейін upload
            upload_capture_to_render(plate, camera_name, image_path)

            print(f"⚠️ EXIT сақталмады: {plate} inside табылмады")


def process_alpr_interval(camera_name, frame):
    """
    Номер 2 рет бірдей оқылса ғана базаға сақтайды.
    Бірақ тез жұмыс істеуі үшін растау уақыты қысқа.
    """
    plate = read_plate_from_frame(frame)

    if not is_good_plate(plate):
        # Номер оқылмаса, бұрынғы жақсы номерді бірден өшірмейміз.
        # Әйтпесе камерада номер көрініп тұрса да экранда --- болып жыпылықтай береді.
        return

    now = time.time()

    # Егер жаңа номер оқылса — растауды қайта бастаймыз
    if plate != plate_confirm[camera_name].get("last", ""):
        plate_confirm[camera_name]["last"] = plate
        plate_confirm[camera_name]["count"] = 1
        plate_confirm[camera_name]["time"] = now

        with state_lock:
            if camera_name == "entry":
                state["entry_plate"] = plate
            else:
                state["exit_plate"] = plate

        print(f"⏳ Растау: {plate} (1/2)")
        return

    # Егер сол номер қайта оқылса — count көбейтеміз
    plate_confirm[camera_name]["count"] += 1
    plate_confirm[camera_name]["time"] = now

    with state_lock:
        if camera_name == "entry":
            state["entry_plate"] = plate
        else:
            state["exit_plate"] = plate

    print(
        f"⏳ Растау: {plate} "
        f"({plate_confirm[camera_name]['count']}/2)"
    )

    # 2 рет сәйкес келсе — базаға сақтаймыз
    if plate_confirm[camera_name]["count"] >= 2:
        save_detected_plate(plate, camera_name, frame)

        plate_confirm[camera_name]["last"] = ""
        plate_confirm[camera_name]["count"] = 0
        plate_confirm[camera_name]["time"] = 0


def run_alpr_async(camera_name, frame):
    """
    ALPR-ды бөлек thread ішінде жүргіземіз.
    Сол кезде камера видеосы тоқтап қалмайды.
    """
    if alpr_busy[camera_name]:
        return

    def job():
        alpr_busy[camera_name] = True
        try:
            process_alpr_interval(camera_name, frame)
        finally:
            alpr_busy[camera_name] = False

    threading.Thread(target=job, daemon=True).start()

# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker(camera_name):
    """
    Камераны бөлек thread ішінде оқиды.
    Браузерге берілетін соңғы кадрды state ішінде ұстайды.
    ALPR әр 1.0 секундта ғана жұмыс істейді.
    """
    print(f"🎥 Camera worker started: {camera_name}")

    cap = None
    last_alpr_time = 0

    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = open_camera(camera_name)

                if cap is None:
                    with state_lock:
                        state[f"{camera_name}_connected"] = False
                        state[f"{camera_name}_message"] = (
                            "Laptop kamera kosylmady"
                            if camera_name == "entry"
                            else "DroidCam kosylmady"
                        )
                        state[f"{camera_name}_frame"] = no_signal_frame(
                            state[f"{camera_name}_message"]
                        )

                    time.sleep(1.5)
                    continue
                
            for _ in range(2):
                cap.grab()

            ok, frame = cap.retrieve()

            if not ok or frame is None or frame.size == 0:
                ok, frame = cap.read()

            if not ok or frame is None or frame.size == 0:
                print(f"⚠️ {camera_name} кадр бермеді, reconnect...")

                try:
                    cap.release()
                except Exception:
                    pass

                cap = None

                with state_lock:
                    state[f"{camera_name}_connected"] = False
                    state[f"{camera_name}_message"] = "Kamera kadr bermedi"
                    state[f"{camera_name}_frame"] = no_signal_frame(
                        "Kamera kadr bermedi"
                    )

                time.sleep(1)
                continue

            # Кадрды жеңілдету: stream үшін overlay саламыз
            with state_lock:
                current_plate = (
                    state["entry_plate"]
                    if camera_name == "entry"
                    else state["exit_plate"]
                )

            display_frame = frame.copy()
            display_frame = draw_overlay(display_frame, camera_name, current_plate)

            with state_lock:
                state[f"{camera_name}_connected"] = True
                state[f"{camera_name}_message"] = "OK"
                state[f"{camera_name}_frame"] = display_frame

            # ALPR әр 1 секундта ғана
            now = time.time()
            if now - last_alpr_time >= 1.0:
                last_alpr_time = now
                run_alpr_async(camera_name, frame.copy())

            time.sleep(0.005)

        except Exception as e:
            print(f"❌ camera_worker {camera_name} қатесі:", e)

            try:
                if cap:
                    cap.release()
            except Exception:
                pass

            cap = None
            time.sleep(1)


def start_camera_worker(camera_name):
    if IS_RENDER:
        return

    if not ENABLE_CAMERA:
        return

    if workers_started[camera_name]:
        return

    workers_started[camera_name] = True

    thread = threading.Thread(
        target=camera_worker,
        args=(camera_name,),
        daemon=True,
    )

    worker_threads[camera_name] = thread
    thread.start()


def start_all_camera_workers():
    start_camera_worker("entry")
    start_camera_worker("exit")


def generate_frames(camera_name, remote=False):
    """
    Бұл функция енді камераны өзі ашпайды.
    Тек background worker дайындаған соңғы кадрды береді.
    Сондықтан браузердегі видео қатпайды.
    """
    if camera_name not in ["entry", "exit"]:
        return

    if not IS_RENDER:
        start_camera_worker(camera_name)

    while True:
        with state_lock:
            frame = state.get(f"{camera_name}_frame")
            message = state.get(f"{camera_name}_message", "Камера күтілуде")

        if frame is None:
            frame = no_signal_frame(message)

        yield encode_frame(frame, remote=remote)

        if remote:
            time.sleep(0.18)
        else:
            time.sleep(0.02)
            
# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("login", "").strip()
        password = request.form.get("password", "").strip()

        db = get_db()
        user = db.execute(
            """
            SELECT *
            FROM users
            WHERE login = ?
              AND password = ?
            """,
            (username, password),
        ).fetchone()
        db.close()

        if user:
            session["logged_in"] = True
            session["login"] = username
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

    rows = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        ORDER BY id DESC
        """
    ).fetchall()

    updated_logs = []

    for row in rows:
        log = dict(row)

        minutes, amount = calculate_amount_by_time(
            log.get("entry_time"),
            log.get("exit_time"),
        )

        # Егер тегін рұқсат болса немесе төленген болса, соманы бұзбаймыз
        if int(log.get("paid") or 0) == 1 and int(log.get("amount") or 0) == 0:
            amount = 0

        log["live_minutes"] = minutes
        log["live_amount"] = amount

        db.execute(
            """
            UPDATE vehicle_logs
            SET duration_minutes = ?,
                amount = ?
            WHERE id = ?
            """,
            (
                minutes,
                amount,
                log["id"],
            ),
        )

        updated_logs.append(log)

    db.commit()
    db.close()

    return render_template("journal.html", logs=updated_logs)


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


@app.route("/event/delete/<int:event_id>", methods=["POST"])
def delete_event(event_id):
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()

    row = db.execute(
        "SELECT * FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()

    if row:
        db.execute(
            "DELETE FROM events WHERE id = ?",
            (event_id,),
        )
        db.commit()

        add_audit(
            row.get("plate") if isinstance(row, dict) else row["plate"],
            "Удалить событие",
            str(dict(row)),
            "",
            "Событие өшірілді",
        )

    db.close()

    return redirect(url_for("events"))


@app.route("/audit")
def audit():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    logs = db.execute(
        """
        SELECT *
        FROM audit_logs
        ORDER BY id DESC
        """
    ).fetchall()
    db.close()

    return render_template("audit.html", logs=logs)


@app.route("/payments")
def payments():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()

    payments_data = db.execute(
        """
        SELECT *
        FROM payments
        ORDER BY id DESC
        """
    ).fetchall()

    lists = db.execute(
        """
        SELECT *
        FROM lists
        ORDER BY id DESC
        """
    ).fetchall()

    balances = db.execute(
        """
        SELECT *
        FROM balances
        ORDER BY id DESC
        """
    ).fetchall()

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
    rows = db.execute(
        """
        SELECT *
        FROM abonements
        ORDER BY id DESC
        """
    ).fetchall()
    db.close()

    return render_template("abonement.html", abonements=rows)


@app.route("/administration")
def administration():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()
    rows = db.execute(
        """
        SELECT *
        FROM administration_access
        ORDER BY id DESC
        """
    ).fetchall()
    db.close()

    return render_template("administration.html", staff=rows)


@app.route("/client")
def client():
    return render_template("client.html")


@app.route("/qr")
def qr_page():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("qr_links.html")


@app.route("/pay")
def pay_page():
    return render_template("qr_pay.html")


# ============================================================
# VIDEO ROUTES
# ============================================================

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


# ============================================================
# QR PAYMENT API
# ============================================================

@app.route("/api/qr-check", methods=["POST"])
def qr_check():
    plate = normalize_plate(request.form.get("plate", ""))

    if not plate:
        return jsonify({"ok": False, "message": "Номер енгізіңіз"})

    db = get_db()

    logs = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
          AND COALESCE(paid, 0) = 0
        ORDER BY id DESC
        """,
        (plate,),
    ).fetchall()

    balance_row = db.execute(
        """
        SELECT *
        FROM balances
        WHERE plate = ?
        """,
        (plate,),
    ).fetchone()

    db.close()

    if not logs:
        return jsonify(
            {
                "ok": False,
                "message": "Номер журналда табылмады немесе қарыз жоқ",
            }
        )

    total_amount = 0
    total_minutes = 0

    for log in logs:
        minutes, amount = calculate_amount_by_time(
            log["entry_time"],
            log["exit_time"],
        )

        total_amount += int(amount or 0)
        total_minutes += int(minutes or 0)

    balance = 0

    if balance_row:
        balance = int(balance_row["balance"] or 0)

    debt = max(total_amount - balance, 0)

    return jsonify(
        {
            "ok": True,
            "plate": plate,
            "amount": debt,
            "debt": debt,
            "balance": balance,
            "minutes": total_minutes,
            "message": f"Жалпы тұрған уақыт: {total_minutes} минут. Төлеу керек: {debt} ₸",
        }
    )


@app.route("/api/qr-pay", methods=["POST"])
def qr_pay():
    plate = normalize_plate(request.form.get("plate", ""))

    if not plate:
        return jsonify({"ok": False, "message": "Номер жоқ"})

    db = get_db()

    unpaid_logs = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
          AND COALESCE(paid, 0) = 0
        ORDER BY id DESC
        """,
        (plate,),
    ).fetchall()

    if not unpaid_logs:
        db.close()
        return jsonify({"ok": False, "message": "Төленбеген қарыз табылмады"})

    total_amount = 0

    for log in unpaid_logs:
        _, amount = calculate_amount_by_time(
            log["entry_time"],
            log["exit_time"],
        )
        total_amount += int(amount or 0)

    balance_row = db.execute(
        "SELECT * FROM balances WHERE plate = ?",
        (plate,),
    ).fetchone()

    balance = int(balance_row["balance"] or 0) if balance_row else 0
    pay_amount = max(total_amount - balance, 0)

    db.execute(
        """
        INSERT INTO payments(plate, amount, method, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            plate,
            pay_amount,
            "QR DEMO",
            now_str(),
        ),
    )

    db.execute(
        """
        UPDATE vehicle_logs
        SET paid = 1
        WHERE plate = ?
          AND COALESCE(paid, 0) = 0
        """,
        (plate,),
    )

    if balance > 0:
        db.execute(
            """
            UPDATE balances
            SET balance = ?
            WHERE plate = ?
            """,
            (max(balance - total_amount, 0), plate),
        )

    db.commit()
    db.close()

    add_audit(
        plate,
        "QR төлем",
        "",
        f"{pay_amount} ₸",
        "Demo payment",
    )

    return jsonify(
        {
            "ok": True,
            "amount": pay_amount,
        }
    )
    
# ============================================================
# GENERAL API ROUTES
# ============================================================

@app.route("/api/last-plates")
def api_last_plates():
    with state_lock:
        entry = state["entry_plate"]
        exit_plate = state["exit_plate"]

    return jsonify(
        {
            "entry": entry,
            "exit": exit_plate,
        }
    )


@app.route("/api/system-status")
def api_system_status():
    with state_lock:
        data = {
            "entry_connected": state["entry_connected"],
            "exit_connected": state["exit_connected"],
            "entry_message": state["entry_message"],
            "exit_message": state["exit_message"],
            "entry_plate": state["entry_plate"],
            "exit_plate": state["exit_plate"],
        }

    return jsonify({"ok": True, **data})


@app.route("/api/manual-entry", methods=["POST"])
def manual_entry():
    plate = normalize_plate(request.form.get("plate", ""))
    direction = request.form.get("direction")

    if not plate:
        return jsonify({"ok": False, "message": "Номер жазыңыз"})

    free, free_type = has_free_access(plate)

    if direction == "entry":
        saved = add_entry(plate, "Қолмен кіргізілді", None)

        if saved and free:
            mark_last_log_free(plate, free_type)

        add_audit(
            plate,
            "Manual entry",
            "",
            plate,
            f"Қолмен кіргізілді | {free_type if free else 'Ақылы'}",
        )

        with state_lock:
            state["entry_plate"] = plate

    elif direction == "exit":
        saved = add_exit(plate, "Қолмен шығарылды", None)

        if saved and free:
            mark_last_log_free(plate, free_type)

        add_audit(
            plate,
            "Manual exit",
            "",
            plate,
            f"Қолмен шығарылды | {free_type if free else 'Ақылы'}",
        )

        with state_lock:
            state["exit_plate"] = plate

        if not saved:
            return jsonify(
                {
                    "ok": False,
                    "message": "Бұл номер ішкі журналда табылмады",
                }
            )

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

    if action == "open":
        barriers[name]["state"] = "open"

    elif action == "close":
        barriers[name]["state"] = "closed"

    elif action == "fix_open":
        barriers[name]["state"] = "open"
        barriers[name]["fixed_open"] = True
        barriers[name]["fixed_close"] = False

    elif action == "fix_close":
        barriers[name]["state"] = "closed"
        barriers[name]["fixed_close"] = True
        barriers[name]["fixed_open"] = False

    elif action == "unfix":
        barriers[name]["fixed_open"] = False
        barriers[name]["fixed_close"] = False

    else:
        return jsonify({"ok": False, "message": "Action қате"})

    fixed_state = "none"

    if barriers[name]["fixed_open"]:
        fixed_state = "open"

    if barriers[name]["fixed_close"]:
        fixed_state = "closed"

    db = get_db()
    db.execute(
        """
        UPDATE barriers
        SET state = ?,
            fixed_state = ?,
            updated_at = ?
        WHERE name = ?
        """,
        (
            barriers[name]["state"],
            fixed_state,
            now_str(),
            name,
        ),
    )
    db.commit()
    db.close()

    add_audit(
        name,
        f"Barrier {action}",
        "",
        barriers[name]["state"],
        reason,
    )

    return jsonify({"ok": True})


# ============================================================
# ABONEMENT / ADMINISTRATION
# ============================================================

@app.route("/api/abonement", methods=["POST"])
def api_abonement():
    plate = normalize_plate(request.form.get("plate", ""))
    full_name = request.form.get("full_name", "").strip()
    phone = request.form.get("phone", "").strip()
    group_name = request.form.get("group_name", "").strip()

    if not plate or not full_name or not phone:
        return jsonify(
            {
                "ok": False,
                "message": "Аты-жөні, телефон, номер міндетті",
            }
        )

    now = datetime.now()
    start_date = now.strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")
    created_at = now_str()

    db = get_db()
    db.execute(
        """
        INSERT INTO abonements(
            full_name,
            phone,
            plate,
            group_name,
            price,
            paid,
            start_date,
            end_date,
            active,
            created_at
        )
        VALUES (?, ?, ?, ?, 5000, 0, ?, ?, 1, ?)
        ON CONFLICT (plate)
        DO UPDATE SET
            full_name = excluded.full_name,
            phone = excluded.phone,
            group_name = excluded.group_name,
            price = 5000,
            start_date = excluded.start_date,
            end_date = excluded.end_date,
            active = 1,
            created_at = excluded.created_at
        """,
        (
            full_name,
            phone,
            plate,
            group_name,
            start_date,
            end_date,
            created_at,
        ),
    )
    db.commit()
    db.close()

    add_audit(
        plate,
        "Абонемент қосылды",
        "",
        f"{full_name} | 5000 ₸ | {end_date}",
        "1 айлық абонемент",
    )

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

    db.execute(
        """
        INSERT INTO payments(plate, amount, method, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (plate, 5000, "Абонемент", now_str()),
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
        return jsonify(
            {
                "ok": False,
                "message": "Барлық мәліметті толтырыңыз",
            }
        )

    db = get_db()
    db.execute(
        """
        INSERT INTO administration_access(
            full_name,
            phone,
            plate,
            position,
            active,
            created_at
        )
        VALUES (?, ?, ?, ?, 1, ?)
        ON CONFLICT (plate)
        DO UPDATE SET
            full_name = excluded.full_name,
            phone = excluded.phone,
            position = excluded.position,
            active = 1,
            created_at = excluded.created_at
        """,
        (
            full_name,
            phone,
            plate,
            position,
            now_str(),
        ),
    )
    db.commit()
    db.close()

    add_audit(
        plate,
        "Администрация access",
        "",
        f"{full_name} | {position}",
        "Тегін кіру рұқсаты",
    )

    return jsonify({"ok": True})


@app.route("/api/free-access/delete", methods=["POST"])
def delete_free_access():
    table = request.form.get("table")
    plate = normalize_plate(request.form.get("plate", ""))

    if table not in ["abonements", "administration_access"]:
        return jsonify({"ok": False, "message": "Кесте қате"})

    db = get_db()
    db.execute(
        f"UPDATE {table} SET active = 0 WHERE plate = ?",
        (plate,),
    )
    db.commit()
    db.close()

    add_audit(
        plate,
        "Free access disabled",
        table,
        "active=0",
        "Рұқсат өшірілді",
    )

    return jsonify({"ok": True})


# ============================================================
# JOURNAL EDIT API
# ============================================================

@app.route("/api/journal/edit-plate", methods=["POST"])
def edit_plate():
    old_plate = normalize_plate(request.form.get("old_plate", ""))
    new_plate = normalize_plate(request.form.get("new_plate", ""))

    if not old_plate or not new_plate:
        return jsonify({"ok": False, "message": "Номер дұрыс емес"})

    db = get_db()

    tables = [
        "vehicle_logs",
        "events",
        "payments",
        "balances",
        "abonements",
        "administration_access",
        "lists",
    ]

    for table in tables:
        db.execute(
            f"UPDATE {table} SET plate = ? WHERE plate = ?",
            (new_plate, old_plate),
        )

    db.commit()
    db.close()

    add_audit(
        new_plate,
        "Изменить номер",
        old_plate,
        new_plate,
        f"{old_plate} → {new_plate}",
    )

    return jsonify({"ok": True})


@app.route("/api/journal/edit-time", methods=["POST"])
def edit_time():
    log_id = request.form.get("log_id")
    entry_time = request.form.get("entry_time")
    exit_time = request.form.get("exit_time")

    if not exit_time:
        exit_time = None

    db = get_db()

    old = db.execute(
        "SELECT * FROM vehicle_logs WHERE id = ?",
        (log_id,),
    ).fetchone()

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
        (
            entry_time,
            exit_time,
            minutes,
            amount,
            status,
            paid,
            note,
            log_id,
        ),
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

    row = db.execute(
        "SELECT * FROM vehicle_logs WHERE id = ?",
        (log_id,),
    ).fetchone()

    if not row:
        db.close()
        return jsonify({"ok": False, "message": "Журнал табылмады"})

    db.execute(
        """
        UPDATE vehicle_logs
        SET amount = 0,
            paid = 1,
            note = ?
        WHERE id = ?
        """,
        ("Қарыз өшірілді", log_id),
    )

    db.commit()
    db.close()

    add_audit(
        row["plate"],
        "Удалить задолженность",
        str(row["amount"]),
        "0",
        "Қарыз өшірілді",
    )

    return jsonify({"ok": True})


@app.route("/api/journal/delete-log", methods=["POST"])
def delete_log():
    log_id = request.form.get("log_id")

    db = get_db()

    row = db.execute(
        "SELECT * FROM vehicle_logs WHERE id = ?",
        (log_id,),
    ).fetchone()

    if not row:
        db.close()
        return jsonify({"ok": False, "message": "Журнал табылмады"})

    db.execute(
        "DELETE FROM vehicle_logs WHERE id = ?",
        (log_id,),
    )

    db.commit()
    db.close()

    add_audit(
        row["plate"],
        "Удалить из журнала",
        str(dict(row)),
        "",
        "Журналдан өшірілді",
    )

    return jsonify({"ok": True})


# ============================================================
# PAYMENTS / LISTS / BALANCE API
# ============================================================

@app.route("/api/list", methods=["POST"])
def api_list():
    plate = normalize_plate(request.form.get("plate", ""))
    list_type = request.form.get("type")
    reason = request.form.get("reason", "")

    if not plate or list_type not in ["white", "black"]:
        return jsonify({"ok": False, "message": "Номер немесе тип қате"})

    db = get_db()
    db.execute(
        """
        INSERT INTO lists(plate, type, reason, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (plate)
        DO UPDATE SET
            type = excluded.type,
            reason = excluded.reason,
            created_at = excluded.created_at
        """,
        (
            plate,
            list_type,
            reason,
            now_str(),
        ),
    )
    db.commit()
    db.close()

    add_audit(
        plate,
        f"Add {list_type} list",
        "",
        list_type,
        reason,
    )

    return jsonify({"ok": True})


@app.route("/api/payment", methods=["POST"])
def api_payment():
    plate = normalize_plate(request.form.get("plate", ""))
    amount = int(request.form.get("amount", 0))
    method = request.form.get("method", "Kaspi")

    if not plate or amount <= 0:
        return jsonify({"ok": False, "message": "Номер немесе сома қате"})

    db = get_db()

    db.execute(
        """
        INSERT INTO payments(plate, amount, method, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            plate,
            amount,
            method,
            now_str(),
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
        "Payment",
        "",
        f"{amount} ₸",
        method,
    )

    return jsonify({"ok": True})


@app.route("/api/balance", methods=["POST"])
def api_balance():
    plate = normalize_plate(request.form.get("plate", ""))
    amount = int(request.form.get("amount", 0))
    operation = request.form.get("operation")

    if not plate or amount <= 0:
        return jsonify({"ok": False, "message": "Номер немесе сома қате"})

    if operation == "minus":
        amount = -amount

    db = get_db()

    db.execute(
        """
        INSERT INTO balances(plate, balance)
        VALUES (?, 0)
        ON CONFLICT (plate)
        DO NOTHING
        """,
        (plate,),
    )

    db.execute(
        """
        UPDATE balances
        SET balance = balance + ?
        WHERE plate = ?
        """,
        (
            amount,
            plate,
        ),
    )

    db.commit()
    db.close()

    add_audit(
        plate,
        "Balance",
        "",
        str(amount),
        operation,
    )

    return jsonify({"ok": True})


# ============================================================
# CLIENT STATUS API
# ============================================================

@app.route("/api/client-status", methods=["POST"])
def client_status():
    plate = normalize_plate(request.form.get("plate", ""))
    messages = []

    if not plate:
        return jsonify(
            {
                "ok": True,
                "plate": plate,
                "messages": ["Номер енгізілмеді."],
            }
        )

    db = get_db()

    unpaid_logs = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
          AND COALESCE(paid, 0) = 0
        ORDER BY id DESC
        """,
        (plate,),
    ).fetchall()

    any_log = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (plate,),
    ).fetchone()

    listed = db.execute(
        "SELECT * FROM lists WHERE plate = ?",
        (plate,),
    ).fetchone()

    balance = db.execute(
        "SELECT * FROM balances WHERE plate = ?",
        (plate,),
    ).fetchone()

    db.close()

    free, free_type = has_free_access(plate)

    if free:
        messages.append(f"Сізде тегін кіру рұқсаты бар: {free_type}.")
        messages.append("Қарыз жоқ.")

    elif not any_log:
        messages.append("Сіздің номеріңіз журналда табылмады.")

    elif unpaid_logs:
        total_amount = 0
        total_minutes = 0

        for log in unpaid_logs:
            minutes, amount = calculate_amount_by_time(
                log["entry_time"],
                log["exit_time"],
            )

            total_amount += int(amount or 0)
            total_minutes += int(minutes or 0)

        messages.append(f"Төлеу керек сома: {total_amount} ₸")
        messages.append(f"Жалпы тұрған уақыт: {total_minutes} минут")

    else:
        messages.append("Төлем жасалған. Қарыз жоқ.")

    if listed:
        if listed["type"] == "black":
            messages.append("Сіздің номеріңіз қара тізімде тұр.")
        elif listed["type"] == "white":
            messages.append("Сіздің номеріңіз ақ тізімде. Тегін өте аласыз.")

    if balance:
        messages.append(f"Сіздің баланс: {balance['balance']} ₸")

    return jsonify(
        {
            "ok": True,
            "plate": plate,
            "messages": messages,
        }
    )


# ============================================================
# CLOUD / REMOTE SYNC
# ============================================================

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
        return {
            "ok": True,
            "plate": plate,
            "direction": "exit",
            "saved": ok,
        }

    ok = add_entry(plate, post, image_path)
    return {
        "ok": True,
        "plate": plate,
        "direction": "entry",
        "saved": ok,
    }


@app.route("/api/remote-entry", methods=["POST"])
def remote_entry():
    secret = request.form.get("secret", "")

    if secret != os.getenv("REMOTE_SECRET", "auezov-secret-2026"):
        return jsonify({"ok": False, "message": "Құпия кілт қате"}), 403

    plate = normalize_plate(request.form.get("plate", ""))
    direction = request.form.get("direction", "entry")
    camera = request.form.get("camera", "Гл корпус кіріс")

    if not plate:
        return jsonify({"ok": False, "message": "Номер жоқ"}), 400

    image_path = None

    if "image" in request.files:
        image = request.files["image"]

        if image and image.filename:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

            filename = f"remote_{direction}_{plate}_{int(time.time())}.jpg"
            save_path = UPLOAD_DIR / filename

            image.save(str(save_path))
            image_path = f"/static/captures/{filename}"

    if direction == "entry":
        saved = add_entry(plate, camera, image_path)

    elif direction == "exit":
        saved = add_exit(plate, camera, image_path)

    else:
        return jsonify({"ok": False, "message": "direction қате"}), 400

    return jsonify(
        {
            "ok": True,
            "plate": plate,
            "direction": direction,
            "saved": saved,
            "image": image_path,
        }
    )


@app.route("/api/upload-capture", methods=["POST"])
def upload_capture():
    secret = request.form.get("secret", "")

    if secret != os.getenv("REMOTE_SECRET", "auezov-secret-2026"):
        return jsonify({"ok": False, "message": "Құпия кілт қате"}), 403

    plate = normalize_plate(request.form.get("plate", ""))
    camera_name = request.form.get("camera_name", "entry")

    if not plate:
        return jsonify({"ok": False, "message": "Номер жоқ"}), 400

    if "image" not in request.files:
        return jsonify({"ok": False, "message": "Сурет жоқ"}), 400

    image = request.files["image"]

    if not image or not image.filename:
        return jsonify({"ok": False, "message": "Файл аты жоқ"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"cloud_{camera_name}_{plate}_{int(time.time())}.jpg"
    save_path = UPLOAD_DIR / filename

    image.save(str(save_path))

    image_url = f"/static/captures/{filename}"

    db = get_db()

    latest_event = db.execute(
        """
        SELECT id
        FROM events
        WHERE plate = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (plate,),
    ).fetchone()

    if latest_event:
        db.execute(
            """
            UPDATE events
            SET image_path = ?
            WHERE id = ?
            """,
            (image_url, latest_event["id"]),
        )

    if camera_name == "entry":
        db.execute(
            """
            UPDATE vehicle_logs
            SET entry_image = ?
            WHERE id = (
                SELECT id
                FROM vehicle_logs
                WHERE plate = ?
                ORDER BY id DESC
                LIMIT 1
            )
            """,
            (image_url, plate),
        )
    else:
        db.execute(
            """
            UPDATE vehicle_logs
            SET exit_image = ?
            WHERE id = (
                SELECT id
                FROM vehicle_logs
                WHERE plate = ?
                ORDER BY id DESC
                LIMIT 1
            )
            """,
            (image_url, plate),
        )

    db.commit()
    db.close()

    return jsonify(
        {
            "ok": True,
            "plate": plate,
            "image_url": image_url,
        }
    )

# ============================================================
# ADMIN / HEALTH
# ============================================================

@app.route("/health")
def health():
    db = get_db()

    try:
        row = db.execute(
            """
            SELECT
                current_database() AS database_name,
                current_user AS database_user,
                inet_server_addr() AS server_ip
            """
        ).fetchone()

        db_info = dict(row) if row else {}

    except Exception as e:
        db_info = {"error": str(e)}

    finally:
        db.close()

    return jsonify(
        {
            "ok": True,
            "app": "Auezov Parking",
            "db": "PostgreSQL" if DATABASE_URL else "SQLite fallback",
            "database_url_exists": bool(DATABASE_URL),
            "camera_enabled": ENABLE_CAMERA,
            "is_render": IS_RENDER,
            "db_info": db_info,
        }
    )


@app.route("/admin/clear-all")
def clear_all_online():
    if not login_required():
        return redirect(url_for("login"))

    db = get_db()

    tables = [
        "vehicle_logs",
        "events",
        "audit_logs",
    ]

    for table in tables:
        db.execute(f"DELETE FROM {table}")

    db.commit()
    db.close()

    add_audit(
        "SYSTEM",
        "Clear demo data",
        "",
        "",
        "Журнал, события, аудит тазаланды",
    )

    return "✅ DATABASE CLEARED"


# ============================================================
# APP START
# ============================================================

if __name__ == "__main__":
    if ENABLE_CAMERA and not IS_RENDER:
        start_all_camera_workers()

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000,
        threaded=True,
        use_reloader=False,
    )