// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-phase-orchestrator — studio-web Vite build (portal + MF remote)

/// <reference types="vitest/config" />
import { federation } from "@module-federation/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { moduleFederationConfig } from "./module-federation.config";

export default defineConfig({
  // Served under /studios/scpn-phase-orchestrator/, so every asset and the
  // remoteEntry resolve there (the platform's Caddy strips the prefix back onto
  // disk). This MUST equal the studio id path per the hosting contract.
  base: "/studios/scpn-phase-orchestrator/",
  plugins: [react(), federation({ ...moduleFederationConfig })],
  build: {
    target: "esnext",
    // The committed evidence snapshot is inlined at build time so the served
    // panel needs no API.
    assetsInlineLimit: 0,
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.{ts,tsx}"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/main.tsx", "src/**/*.test.{ts,tsx}"],
      thresholds: {
        statements: 100,
        branches: 100,
        functions: 100,
        lines: 100,
      },
    },
  },
});
