"""Phase 9: the actual current-season forecast - the payoff of everything
built so far. Thin CLI wrapper around src/live_forecast.py's
build_current_forecast(), which is the single reusable pipeline this
script and the FastAPI backend (Phase 11) both call - see that module's
docstring for the full reasoning.
"""

from pathlib import Path

from src.live_forecast import build_current_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    forecast = build_current_forecast()

    print(f"[live_forecast] {forecast.season}: {forecast.n_played} played, "
          f"{forecast.n_remaining} remaining fixtures")
    print(f"[live_forecast] training complete, generated at {forecast.generated_at}")

    print(f"\n=== {forecast.season} forecast ({forecast.n_simulations:,} simulated seasons) ===")
    cols = ["title_probability", "champions_league_probability",
            "relegation_probability", "expected_position", "expected_points"]
    print(forecast.summary[cols].to_string(float_format=lambda x: f"{x:.3f}"))

    out_path = PROJECT_ROOT / "experiments" / "live_forecast_2026_27.csv"
    forecast.summary.to_csv(out_path)
    print(f"\n[live_forecast] wrote -> {out_path}")


if __name__ == "__main__":
    main()
