# src/utils/emoji_log.py
from colorama import Fore, Style


def info(message: str):
    """💬 Informational message."""
    print(Fore.CYAN + f"💬 {message}" + Style.RESET_ALL)


def success(message: str):
    """✅ Success message."""
    print(Fore.GREEN + f"✅ {message}" + Style.RESET_ALL)


def warn(message: str):
    """⚠️ Warning message."""
    print(Fore.YELLOW + f"⚠️ {message}" + Style.RESET_ALL)


def error(message: str):
    """❌ Error message."""
    print(Fore.RED + f"❌ {message}" + Style.RESET_ALL)


def task(message: str):
    """🚀 Task start or progress."""
    print(Fore.BLUE + f"🚀 {message}" + Style.RESET_ALL)


def done(message: str):
    """🏁 Task completed."""
    print(Fore.MAGENTA + f"🏁 {message}" + Style.RESET_ALL)


def data(message: str):
    """📊 Data-related log."""
    print(Fore.LIGHTBLUE_EX + f"📊 {message}" + Style.RESET_ALL)


def save(message: str):
    """💾 File save operation."""
    print(Fore.LIGHTGREEN_EX + f"💾 {message}" + Style.RESET_ALL)
