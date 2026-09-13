#!/bin/zsh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  echo ""
  echo "❌ .venv bulunamadı."
  echo "İlk kurulum için:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo ""
  read "?Kapatmak için Enter..."
  exit 1
fi

source .venv/bin/activate

if ! command -v streamlit >/dev/null 2>&1; then
  echo ""
  echo "❌ Streamlit bu virtual environment içinde bulunamadı."
  echo "Çalıştır:"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo ""
  read "?Kapatmak için Enter..."
  exit 1
fi

echo "🎵 Chordonomicon Research Explorer başlatılıyor..."
echo "Tarayıcı açılmazsa: http://localhost:8501"
echo ""

exec streamlit run app.py --server.address 127.0.0.1 --server.port 8501
