"""
Application Configuration
InsightGPT - Enterprise AI Data Analyst
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# APP
# =====================================================

APP_NAME = "Northstar"

APP_ICON = "📊"

APP_LAYOUT = "wide"

APP_VERSION = "1.0.0"

# =====================================================
# GEMINI
# =====================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash"

# =====================================================
# FILES
# =====================================================

SUPPORTED_FILES = ["csv"]

MAX_FILE_SIZE_MB = 200

UPLOAD_FOLDER = "uploads"

EXPORT_FOLDER = "exports"

REPORT_FOLDER = "reports"

# =====================================================
# CHARTS
# =====================================================

PLOTLY_TEMPLATE = "plotly_white"

DEFAULT_CHART_HEIGHT = 500

# =====================================================
# DATA
# =====================================================

DEFAULT_PREVIEW_ROWS = 10

MAX_CATEGORY_VALUES = 25

# =====================================================
# COLORS
# =====================================================

PRIMARY_COLOR = "#2563EB"

SUCCESS_COLOR = "#16A34A"

WARNING_COLOR = "#D97706"

DANGER_COLOR = "#DC2626"

# =====================================================
# CREATE DIRECTORIES
# =====================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(EXPORT_FOLDER, exist_ok=True)

os.makedirs(REPORT_FOLDER, exist_ok=True)