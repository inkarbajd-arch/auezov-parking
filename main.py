import threading
import webview
from server import app


def run_flask():
    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    webview.create_window(
        "Auezov Parking",
        "http://127.0.0.1:5000",
        width=1400,
        height=900,
    )

    webview.start()