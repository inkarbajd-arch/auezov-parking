import os
import math
import sqlite3
from datetime import datetime

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_NAME = "parking.db"

PG_POOL = None

if DATABASE_URL:
    PG_POOL = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL,
        cursor_factory=psycopg2.extras.RealDictCursor,
        connect_timeout=20,
        sslmode="require",
    )


class DB:
    """
    DATABASE_URL болса PostgreSQL қолданады.
    DATABASE_URL болмаса SQLite fallback қолданады.
    server.py ішінде ? placeholder қолдансаң да,
    PostgreSQL кезінде автоматты түрде %s қылып ауыстырады.
    """

    def __init__(self):
        self.is_postgres = bool(DATABASE_URL)

        if self.is_postgres:
            if PG_POOL is None:
                raise RuntimeError("PostgreSQL pool дайын емес. DATABASE_URL тексеріңіз.")
            self.conn = PG_POOL.getconn()
        else:
            self.conn = sqlite3.connect(DB_NAME)
            self.conn.row_factory = sqlite3.Row

    def _convert_sql(self, sql: str) -> str:
        if self.is_postgres:
            return sql.replace("?", "%s")
        return sql

    def execute(self, sql, params=None):
        sql = self._convert_sql(sql)
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        return cur

    def executescript(self, sql: str):
        if self.is_postgres:
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for statement in statements:
                self.execute(statement)
        else:
            self.conn.executescript(sql)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        if self.is_postgres:
            PG_POOL.putconn(self.conn)
        else:
            self.conn.close()


def get_db():
    return DB()


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def column_exists(db, table_name, column_name):
    if db.is_postgres:
        row = db.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
              AND column_name = ?
            """,
            (table_name, column_name),
        ).fetchone()
        return row is not None

    rows = db.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


def add_column_if_not_exists(db, table_name, column_name, column_def):
    if not column_exists(db, table_name, column_name):
        db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def migrate_db(db):
    """
    Бұрын жасалған базаға жаңа бағандар керек болса, автоматты қосады.
    """
    add_column_if_not_exists(db, "users", "created_at", "TEXT")

    add_column_if_not_exists(db, "vehicle_logs", "created_at", "TEXT")
    add_column_if_not_exists(db, "vehicle_logs", "entry_image", "TEXT")
    add_column_if_not_exists(db, "vehicle_logs", "exit_image", "TEXT")
    add_column_if_not_exists(db, "vehicle_logs", "note", "TEXT")

    add_column_if_not_exists(db, "barriers", "updated_at", "TEXT")


def init_db():
    db = get_db()

    if db.is_postgres:
        id_type = "SERIAL PRIMARY KEY"
        text_type = "TEXT"
        int_type = "INTEGER"
    else:
        id_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
        text_type = "TEXT"
        int_type = "INTEGER"

    db.executescript(f"""
        CREATE TABLE IF NOT EXISTS users (
            id {id_type},
            login {text_type} UNIQUE NOT NULL,
            password {text_type} NOT NULL,
            created_at {text_type}
        );

        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id {id_type},
            plate {text_type} NOT NULL,
            entry_time {text_type},
            exit_time {text_type},
            entry_post {text_type},
            exit_post {text_type},
            duration_minutes {int_type} DEFAULT 0,
            amount {int_type} DEFAULT 200,
            paid {int_type} DEFAULT 0,
            status {text_type} DEFAULT 'inside',
            entry_image {text_type},
            exit_image {text_type},
            note {text_type},
            created_at {text_type}
        );

        CREATE TABLE IF NOT EXISTS events (
            id {id_type},
            plate {text_type},
            camera {text_type},
            event_type {text_type},
            image_path {text_type},
            created_at {text_type} NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id {id_type},
            plate {text_type},
            action {text_type} NOT NULL,
            old_value {text_type},
            new_value {text_type},
            comment {text_type},
            created_at {text_type} NOT NULL
        );

        CREATE TABLE IF NOT EXISTS payments (
            id {id_type},
            plate {text_type} NOT NULL,
            amount {int_type} NOT NULL,
            method {text_type} NOT NULL,
            created_at {text_type} NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lists (
            id {id_type},
            plate {text_type} UNIQUE NOT NULL,
            type {text_type} NOT NULL,
            reason {text_type},
            created_at {text_type} NOT NULL
        );

        CREATE TABLE IF NOT EXISTS balances (
            id {id_type},
            plate {text_type} UNIQUE NOT NULL,
            balance {int_type} DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS barriers (
            id {id_type},
            name {text_type} UNIQUE NOT NULL,
            state {text_type} DEFAULT 'closed',
            fixed_state {text_type} DEFAULT 'none',
            updated_at {text_type}
        );

        CREATE TABLE IF NOT EXISTS abonements (
            id {id_type},
            full_name {text_type} NOT NULL,
            phone {text_type} NOT NULL,
            plate {text_type} UNIQUE NOT NULL,
            group_name {text_type},
            price {int_type} DEFAULT 5000,
            paid {int_type} DEFAULT 0,
            start_date {text_type} NOT NULL,
            end_date {text_type} NOT NULL,
            active {int_type} DEFAULT 1,
            created_at {text_type} NOT NULL
        );

        CREATE TABLE IF NOT EXISTS administration_access (
            id {id_type},
            full_name {text_type} NOT NULL,
            phone {text_type} NOT NULL,
            plate {text_type} UNIQUE NOT NULL,
            position {text_type} NOT NULL,
            active {int_type} DEFAULT 1,
            created_at {text_type} NOT NULL
        );
    """)

    migrate_db(db)

    db.executescript("""
        CREATE INDEX IF NOT EXISTS idx_vehicle_logs_plate ON vehicle_logs(plate);
        CREATE INDEX IF NOT EXISTS idx_vehicle_logs_status ON vehicle_logs(status);
        CREATE INDEX IF NOT EXISTS idx_vehicle_logs_plate_status ON vehicle_logs(plate, status);
        CREATE INDEX IF NOT EXISTS idx_vehicle_logs_paid ON vehicle_logs(paid);

        CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate);
        CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

        CREATE INDEX IF NOT EXISTS idx_payments_plate ON payments(plate);
        CREATE INDEX IF NOT EXISTS idx_payments_created_at ON payments(created_at);

        CREATE INDEX IF NOT EXISTS idx_audit_plate ON audit_logs(plate);
        CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_logs(created_at);
    """)

    db.execute(
        """
        INSERT INTO users(login, password, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT (login)
        DO UPDATE SET password = excluded.password
        """,
        ("Auezov", "1943", now_str()),
    )

    db.execute(
        """
        INSERT INTO barriers(name, state, fixed_state, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (name)
        DO NOTHING
        """,
        ("entry", "closed", "none", now_str()),
    )

    db.execute(
        """
        INSERT INTO barriers(name, state, fixed_state, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (name)
        DO NOTHING
        """,
        ("exit", "closed", "none", now_str()),
    )

    db.commit()
    db.close()


def add_audit(plate, action, old_value="", new_value="", comment=""):
    db = get_db()

    try:
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
                str(comment) if comment is not None else "",
                now_str(),
            ),
        )
        db.commit()

    except Exception as e:
        db.rollback()
        print("❌ add_audit қатесі:", e)

    finally:
        db.close()


def add_event(plate, camera, event_type, image_path=None):
    db = get_db()

    try:
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
            (
                plate,
                camera,
                event_type,
                image_path,
                now_str(),
            ),
        )
        db.commit()

    except Exception as e:
        db.rollback()
        print("❌ add_event қатесі:", e)

    finally:
        db.close()


def calculate_amount(entry_time, exit_time):
    try:
        if not entry_time or not exit_time:
            return 0, 0

        entry_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
        exit_dt = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")

        minutes = max(int((exit_dt - entry_dt).total_seconds() // 60), 0)

        if minutes <= 120:
            amount = 200
        else:
            extra_minutes = minutes - 120
            extra_hours = math.ceil(extra_minutes / 60)
            amount = 200 + extra_hours * 100

        return minutes, amount

    except Exception as e:
        print("❌ calculate_amount қатесі:", e)
        return 0, 0


def add_entry(plate, post, image_path=None):
    """
    Машина кірген кезде шақырылады.
    Егер сол номер already inside болса, қайталамау үшін False қайтарады.
    """
    db = get_db()
    now = now_str()

    try:
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
                note,
                created_at
            )
            VALUES (?, ?, NULL, ?, NULL, 0, 200, 0, 'inside', ?, NULL, NULL, ?)
            """,
            (
                plate,
                now,
                post,
                image_path,
                now,
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
            VALUES (?, ?, 'entry', ?, ?)
            """,
            (
                plate,
                post,
                image_path,
                now,
            ),
        )

        db.commit()

    except Exception as e:
        db.rollback()
        print("❌ add_entry қатесі:", e)
        db.close()
        return False

    db.close()

    add_audit(
        plate,
        "Кіру",
        "",
        f"{post} | {now}",
        "Машина парковкаға кірді",
    )

    return True


def add_exit(plate, post, image_path=None):
    """
    Машина шыққан кезде шақырылады.
    Егер inside журналда табылмаса False қайтарады.
    """
    db = get_db()
    now = now_str()

    try:
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

        duration_minutes, amount = calculate_amount(row["entry_time"], now)

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
            (
                plate,
                post,
                image_path,
                now,
            ),
        )

        db.commit()

    except Exception as e:
        db.rollback()
        print("❌ add_exit қатесі:", e)
        db.close()
        return False

    db.close()

    add_audit(
        plate,
        "Шығу",
        row["entry_time"],
        f"{post} | {now} | {duration_minutes} минут | {amount} ₸",
        "Машина парковкадан шықты",
    )

    return True


def get_last_logs(limit=50):
    db = get_db()

    rows = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    db.close()
    return rows


def get_unpaid_logs(plate):
    db = get_db()

    rows = db.execute(
        """
        SELECT *
        FROM vehicle_logs
        WHERE plate = ?
          AND COALESCE(paid, 0) = 0
        ORDER BY id DESC
        """,
        (plate,),
    ).fetchall()

    db.close()
    return rows


def mark_log_paid(plate):
    db = get_db()

    try:
        db.execute(
            """
            UPDATE vehicle_logs
            SET paid = 1
            WHERE plate = ?
              AND COALESCE(paid, 0) = 0
            """,
            (plate,),
        )
        db.commit()

    except Exception as e:
        db.rollback()
        print("❌ mark_log_paid қатесі:", e)

    finally:
        db.close()


def reset_all_demo_data():
    """
    Демо үшін журнал/оқиғаларды тазалау.
    users және barriers сақталады.
    """
    db = get_db()

    tables = [
        "vehicle_logs",
        "events",
        "audit_logs",
        "payments",
        "lists",
        "balances",
        "abonements",
        "administration_access",
    ]

    try:
        for table in tables:
            db.execute(f"DELETE FROM {table}")

            if not db.is_postgres:
                try:
                    db.execute(
                        "DELETE FROM sqlite_sequence WHERE name = ?",
                        (table,),
                    )
                except Exception:
                    pass

        db.commit()

    except Exception as e:
        db.rollback()
        print("❌ reset_all_demo_data қатесі:", e)

    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("✅ Database дайын")
    print("✅ Mode:", "PostgreSQL" if DATABASE_URL else "SQLite fallback")