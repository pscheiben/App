from gui import GatingApp

__version__ = "2.3.0"

if __name__ == "__main__":
    print(f"Launching OmniGate SI v{__version__}...")
    app = GatingApp()
    app.mainloop()