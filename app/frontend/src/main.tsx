import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { StandingsPage } from "./pages/StandingsPage";
import { TeamPage } from "./pages/TeamPage";
import "./theme.css";
import "./app.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<StandingsPage />} />
          <Route path="/teams/:team" element={<TeamPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
