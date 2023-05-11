"""
Startup module to load anything that needs to load first for other
things to load so they load when they're supposed to because these
things got loaded first. K?
"""

from dotenv import load_dotenv

load_dotenv()
