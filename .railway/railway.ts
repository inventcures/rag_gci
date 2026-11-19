// Railway infrastructure for the study server: the "api" service and its volume.
//
// Railway stops reading railway.toml and railway.json on 2026-12-01, and new
// services cannot use them, so the API service is declared here instead.
// Railway never reads this file on a push. An operator applies it from a
// terminal linked to the Railway project:
//
//   cd .railway && npm ci && cd ..
//   railway link                 # pick the project and environment
//   railway config plan          # read the plan before applying it
//   railway config apply
//
// Operator guide, with the dashboard steps if you prefer clicking:
// docs/deploy_railway.md. Decisions: docs/decisions/railway-api.md.
//
// This is a partial: it manages only what it declares. The expert review
// console (railway.toml) is a separate service and is left alone.
import { defineRailway, github, preserve, project, service, volume } from "railway/iac";

export const partial = "api";

// Singapore. Railway has no India region (docs.railway.com/deployments/regions),
// so this deployment is for testing with made-up people only. Identifiable
// study data must stay India-hosted.
const REGION = "asia-southeast1-eqsg3a";

export default defineRailway(() => {
  // 5 GB is the Hobby plan maximum and fits the bge-m3 cache (2.3 GB), the
  // vector index and test data. Growing it later is safe; shrinking is not.
  const data = volume("api-data", { region: REGION, sizeMB: 5000 });

  const api = service("api", {
    source: github("inventcures/rag_gci", { branch: "main" }),
    build: { builder: "DOCKERFILE", dockerfilePath: "Dockerfile.api" },
    start: "python scripts/railway_api_start.py",
    // The first boot downloads bge-m3 before the server listens.
    healthcheck: "/health",
    healthcheckTimeout: 900,
    // One replica in one region. A volume cannot be shared, and several
    // processes still keep separate state.
    replicas: { [REGION]: 1 },
    deploy: { restartPolicyType: "ON_FAILURE", restartPolicyMaxRetries: 3 },
    volumeMounts: { "/data": data },
    env: {
      PORT: "8000",
      // Set these in the Railway dashboard (Variables). preserve() keeps the
      // value Railway holds and never writes one into this public file.
      MOBILE_JWT_SECRET: preserve(),
      ADMIN_JWT_SECRET: preserve(),
      GEMINI_API_KEY: preserve(),
      SARVAM_API_KEY: preserve(),
      STUDY_LOG_SALT: preserve(),
    },
  });

  return project("palli-sahayak", { resources: [api, data] });
});
