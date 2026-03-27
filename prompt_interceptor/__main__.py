"""
Entry point for `python -m prompt_interceptor`.
Launches the PromptInterceptor desktop configuration window.
"""
from .launcher import launch


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
