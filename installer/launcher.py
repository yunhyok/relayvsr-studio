"""PyInstaller entry point for the RelayVSR Studio GUI."""

import multiprocessing
import sys

from relayvsr_studio.app.main import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
