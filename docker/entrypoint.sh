set -e

echo "========================================================"
echo "🚀 [ENTRYPOINT] Inicializando o ambiente do projeto..."
echo "========================================================"

# 1. Validar se o arquivo .env ou as variáveis obrigatórias existem
if [ -f "scripts/validate_env.py" ]; then
    echo "📋 [ENTRYPOINT] Validando variáveis de ambiente..."
    python scripts/validate_env.py
else
    echo "⚠️ [ENTRYPOINT] Aviso: scripts/validate_env.py não encontrado. Pulando validação."
fi

export PYTHONPATH="${PYTHONPATH}:/app/src"

echo "📁 [ENTRYPOINT] Verificando estrutura de pastas de dados..."
mkdir -p data/raw data/interim data/processed data/external models notebooks

echo "========================================================"
echo "✅ [ENTRYPOINT] Ambiente pronto! Executando comando final."
echo "========================================================"

exec "$@"