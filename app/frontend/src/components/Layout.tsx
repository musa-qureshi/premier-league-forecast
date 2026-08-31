import { Link, Outlet } from "react-router-dom";

/** App shell: header/nav plus a persistent uncertainty banner. The banner
 * isn't a legal disclaimer bolted on afterward - it's a direct expression
 * of the project's core framing (README "Don't claim the model is
 * correct"): every probability this app shows is a simulated estimate
 * from a model trained on historical data, not a prediction of what will
 * happen, and that has to be visible everywhere the numbers are, not just
 * in a README nobody reading the dashboard will see. */
export function Layout() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <Link to="/" className="app-title">
          Premier League Forecast
        </Link>
        <nav>
          <Link to="/">Standings</Link>
        </nav>
      </header>
      <div className="uncertainty-banner">
        Simulated probabilities from a statistical model, not predictions of what
        will happen — see <a href="https://github.com/musa-qureshi/premier-league-forecast#readme" target="_blank" rel="noreferrer">methodology</a>.
      </div>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
