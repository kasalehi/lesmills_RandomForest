import logging
from pathlib import Path
from datetime import datetime


file_path=Path(__file__).parent.parent / "logs"
file_path.mkdir(exist_ok=True)
log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
log_path = file_path / log_file_name
logging.basicConfig(
    filename=log_path,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger()