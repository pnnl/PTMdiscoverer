import os
import subprocess
import sys

from dotenv import load_dotenv

from ..logger_config import get_logger  # Use centralized logger

logger = get_logger(__name__)


def start_harness_cli():
    """CLI entry point to start the Streamlit test harness."""
    # Load .env for local runs. Docker Compose injects env vars for containers.
    if os.path.exists(".env"):
        load_dotenv()
    elif os.path.exists("../../.env"):
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), "../../.env"))

    app_py_path = os.path.join(os.path.dirname(__file__), "app.py")

    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    address = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")  # noqa: S104 — intentional, Docker listens on all interfaces

    cmd = [
        "streamlit",
        "run",
        app_py_path,
        "--server.port",
        port,
        "--server.address",
        address,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",  # Disable telemetry + external IP lookup
    ]

    logger.info(f"Starting Streamlit harness with command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd)  # noqa: S603 — cmd is built from trusted constants, not user input
        process.wait()
    except FileNotFoundError:
        err_msg = "'streamlit' command not found. Make sure Streamlit is installed and in your PATH."
        logger.critical(err_msg)
        print(f"Error: {err_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error running Streamlit: {e}", exc_info=True)
        print(f"Error running Streamlit: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    start_harness_cli()
