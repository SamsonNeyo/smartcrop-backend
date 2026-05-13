from app.forecast_model import ensure_models


def main() -> None:
    ensure_models()
    print("SARIMA forecast models are ready in backend/models/.")


if __name__ == "__main__":
    main()
