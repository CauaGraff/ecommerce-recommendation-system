import importlib
import sys
from dataclasses import dataclass

from src.config.settings import get_settings

MIN_PYTHON_VERSION = (3, 11)
REQUIRED_PACKAGES = ("torch", "sklearn", "mlflow", "numpy", "pandas", "pydantic_settings")


@dataclass
class CheckResult:
    """Resultado de uma checagem individual do ambiente.

    Attributes:
        name: Nome descritivo da checagem.
        passed: Se a checagem foi bem-sucedida.
        detail: Mensagem adicional (motivo da falha ou valor encontrado).
    """

    name: str
    passed: bool
    detail: str


def check_python_version() -> CheckResult:
    """Verifica se a versão do Python atende ao mínimo exigido.

    Returns:
        O resultado da checagem de versão do Python.
    """
    current = sys.version_info[:2]
    passed = current >= MIN_PYTHON_VERSION
    detail = f"Python {current[0]}.{current[1]} encontrado"
    return CheckResult(name="Versão do Python", passed=passed, detail=detail)


def check_required_packages() -> list[CheckResult]:
    """Verifica se todas as bibliotecas obrigatórias estão instaladas.

    Returns:
        Uma lista com o resultado da checagem de cada pacote requerido.
    """
    results: list[CheckResult] = []
    for package_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package_name)
            results.append(CheckResult(package_name, True, "instalado"))
        except ImportError as error:
            results.append(CheckResult(package_name, False, str(error)))
    return results


def check_settings_load() -> CheckResult:
    """Verifica se as configurações (.env) carregam sem erro.

    Returns:
        O resultado da checagem de carregamento de Settings.
    """
    try:
        settings = get_settings()
        detail = f"model_type={settings.model_type}, seed={settings.random_seed}"
        return CheckResult("Carregamento de Settings", True, detail)
    except Exception as error:  # noqa: BLE001 - queremos capturar qualquer falha aqui
        return CheckResult("Carregamento de Settings", False, str(error))


def print_report(results: list[CheckResult]) -> bool:
    """Imprime o relatório de checagens no console.

    Args:
        results: Lista de resultados a serem exibidos.

    Returns:
        True se todas as checagens passaram, False caso contrário.
    """
    all_passed = True
    for result in results:
        status = "OK " if result.passed else "FALHOU"
        print(f"[{status}] {result.name}: {result.detail}")
        all_passed = all_passed and result.passed
    return all_passed


def main() -> int:
    """Executa todas as checagens e retorna o código de saída do processo.

    Returns:
        0 se todas as checagens passaram, 1 caso alguma tenha falhado.
    """
    results = [check_python_version(), *check_required_packages(), check_settings_load()]
    all_passed = print_report(results)

    if all_passed:
        print("\nAmbiente validado com sucesso.")
        return 0

    print("\nAmbiente incompleto. Rode 'uv sync' e verifique o .env.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
