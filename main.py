from gui import GatingApp

__version__ = "1.1.0"

if __name__ == "__main__":
    print(f"Launching Gating Tool v{__version__} for PCIe Fixture Analysis...")
    app = GatingApp()
    app.mainloop()