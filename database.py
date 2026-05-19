import sqlite3
from datetime import datetime


DB_NAME = "parking.db"


def get_db():
    db = sqlite3.connect(DB_NAME)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()

    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            login TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            entry_time TEXT,
            exit_time TEXT,
            entry_post TEXT,
            exit_post TEXT,
            duration_minutes INTEGER DEFAULT 0,
            amount INTEGER DEFAULT 200,
            paid INTEGER DEFAULT 0,
            status TEXT DEFAULT 'inside',
            entry_image TEXT,
            exit_image TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            camera TEXT,
            event_type TEXT,
            image_path TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            action TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            comment TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            amount INTEGER NOT NULL,
            method TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS balances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT UNIQUE NOT NULL,
            balance INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS barriers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            state TEXT DEFAULT 'closed',
            fixed_state TEXT DEFAULT 'none'
        );

        CREATE TABLE IF NOT EXISTS abonements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            plate TEXT UNIQUE NOT NULL,
            group_name TEXT,
            price INTEGER DEFAULT 5000,
            paid INTEGER DEFAULT 0,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL

        );

        CREATE TABLE IF NOT EXISTS administration_access (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            plate TEXT UNIQUE NOT NULL,
            position TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );
        """
    )

    db.execute(
        "INSERT OR IGNORE INTO users(login, password) VALUES (?, ?)",
        ("Auezov", "1943"),
    )

    db.execute(
        "INSERT OR IGNORE INTO barriers(name, state, fixed_state) VALUES (?, ?, ?)",
        ("entry", "closed", "none"),
    )

    db.execute(
        "INSERT OR IGNORE INTO barriers(name, state, fixed_state) VALUES (?, ?, ?)",
        ("exit", "closed", "none"),
    )

    db.commit()
    db.close()


def add_audit(plate, action, old_value="", new_value="", comment=""):
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.execute(
        """
        INSERT INTO audit_logs(
            plate,
            action,
            old_value,
            new_value,
            comment,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            plate,
            action,
            str(old_value) if old_value is not None else "",
            str(new_value) if new_value is not None else "",
            comment,
            now,
        ),
    )

    db.commit()
    db.close()


def add_event(plate, camera, event_type, image_path=None):
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.execute(
        """
        INSERT INTO events(
            plate,
            camera,
            event_type,
            image_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (plate, camera, event_type, image_path, now),
    )

    db.commit()
    db.close()


def add_entry(plate, post, image_path=None):
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
          AND status = 'inside'
        ORDER BY id DESC
        LIMIT 1
        """,
        (plate,),
    ).fetchone()

    if existing:
        db.close()
        return False

    db.execute(
        """
        INSERT INTO vehicle_logs(
            plate,
            entry_time,
            exit_time,
            entry_post,
            exit_post,
            duration_minutes,
            amount,
            paid,
            status,
            entry_image,
            exit_image,
            note
        )
        VALUES (?, ?, NULL, ?, NULL, 0, 200, 0, 'inside', ?, NULL, NULL)
        """,
        (plate, now, post, image_path),
    )

    db.execute(
        """
        INSERT INTO events(
            plate,
            camera,
            event_type,
            image_path,
            created_at
        )
        VALUES (?, ?, 'entry', ?, ?)
        """,
        (plate, post, image_path, now),
    )

    db.commit()
    db.close()

    add_audit(plate, "Кіру", "", f"{post} | {now}", "Машина парковкаға кірді")

    return True


def add_exit(plate, post, image_path=None):
    db = get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
          AND status = 'inside'
        ORDER BY id DESC
        LIMIT 1
        """,
        (plate,),
    ).fetchone()

    if not row:
        db.close()
        return False

    entry_time = datetime.strptime(row["entry_time"], "%Y-%m-%d %H:%M:%S")
    exit_time = datetime.strptime(now, "%Y-%m-%d %H:%M:%S")

    duration_minutes = max(
        int((exit_time - entry_time).total_seconds() // 60),
        0,
    )

    amount = 200

    if duration_minutes > 120:
        import math

        extra_minutes = duration_minutes - 120
        extra_hours = math.ceil(extra_minutes / 60)
        amount += extra_hours * 100

    if int(row["paid"] or 0) == 1:
        amount = int(row["amount"] or 0)

    db.execute(
        """
        UPDATE vehicle_logs
        SET exit_time = ?,
            exit_post = ?,
            duration_minutes = ?,
            amount = ?,
            status = 'exited',
            exit_image = ?
        WHERE id = ?
        """,
        (
            now,
            post,
            duration_minutes,
            amount,
            image_path,
            row["id"],
        ),
    )

    db.execute(
        """
        INSERT INTO events(
            plate,
            camera,
            event_type,
            image_path,
            created_at
        )
        VALUES (?, ?, 'exit', ?, ?)
        """,
        (plate, post, image_path, now),
    )

    db.commit()
    db.close()

    add_audit(plate, "Шығу", row["entry_time"], f"{post} | {now}", "Машина парковкадан шықты")

    return True


if __name__ == "__main__":
    init_db()
    print("✅ Database дайын")