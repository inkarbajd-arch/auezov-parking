import threading
import time
import webview
from server import app
from database import init_db


def run_flask():
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    time.sleep(2)

    webview.create_window(
        "Auezov Parking",
        "http://127.0.0.1:5000",
        width=1400,
        height=850,
        resizable=True,
        fullscreen=False,
    )

    webview.start()