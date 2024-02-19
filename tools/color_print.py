# Try import colorama, if not, create a 'dummy' environment
_USE_COLORAMA = False
try:
    import colorama

    _USE_COLORAMA = True

    RED = colorama.Fore.RED
    GREEN = colorama.Fore.GREEN
    YELLOW = colorama.Fore.YELLOW
    CYAN = colorama.Fore.CYAN
except ModuleNotFoundError:
    RED = ""
    GREEN = ""
    YELLOW = ""
    CYAN = ""

def init():
    if _USE_COLORAMA:
        colorama.init(autoreset=True)
    else:
        print("'colorama' is not installed. Print will not be colored.")
