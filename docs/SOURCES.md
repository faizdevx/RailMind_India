# Sources and what they support

Research performed 2026-10-12 via web search and page fetches. Summaries are of what was
retrieved; nothing was verified beyond that. **The simulator's rules (block occupancy, 2-min
headway, loop capacity 3, 20-min hold cap, priority weights) are modelling assumptions of this
prototype; no source below specifies those numbers.**

| Source | Issuer / type | Retrieved | Supports |
|---|---|---|---|
| [Indian Railways Public Accounts Committee report on train punctuality (18_Public_Accounts_36)](https://eparlib.sansad.in/bitstream/123456789/3017115/1/18_Public_Accounts_36.pdf) | Parliament of India, Public Accounts Committee (search-result summary; full PDF not read end to end) | 2026-10-12 | Punctuality measured at terminating stations; 15-minute tolerance versus schedule; reported punctuality 90.48 % (2021-22), 73.62 % (2023-24), 78.67 % (2024-25 partial). Basis for reporting delay vs schedule at the corridor end. |
| [Dataful: Train punctuality, Ministry of Railways dataset](https://dataful.in/datasets/1204/) | Aggregator of Ministry of Railways data | 2026-10-12 | Punctuality defined for Mail/Express trains arriving on right time at destination. Aggregate only; no per-train history. |
| [indianrail.gov.in NTES enquiry page](https://www.indianrail.gov.in/enquiry/StaticPages/StaticEnquiry.jsp?StaticPage=index.html) | CRIS / Indian Railways | 2026-10-12 | NTES is the official public train-enquiry portal. The fetched page lists **no public API, terms of use or data-access policy**. Scraping it is therefore **not used** here. |
| [IRTMM chapter 4 (indianrailways.gov.in)](https://indianrailways.gov.in/railwayboard/uploads/codesmanual/IRTMM-RDSO/INDEX4.html) | Indian Railways (track machine manual) | 2026-10-12 | Only that block working / single-line vs double-line movement rules exist as formal rules. Not used for numeric parameters. |
| [ECR Section Controller syllabus (PDF)](https://ecr.indianrailways.gov.in/cris//uploads/files/1718891582418-SECTION%20CONTROLLER%20SYLLABUS-4.pdf) | East Central Railway | 2026-10-12 | That train controllers manage movement from a control office (training topics only). No rule text retrieved. |
| [IRFCA General Rules mirror](https://www.irfca.org/docs/rulebook/gr-stations.html) | Community site, **not official** | 2026-10-12 | Station working under General Rules; line-clear / block concepts. Context only. |

## Provider research

| Candidate | Finding | Status in this project |
|---|---|---|
| NTES / indianrail.gov.in | Official, but no documented public API or data terms found. | Not used (no authorised programmatic access found). |
| RapidAPI "IRCTC" family (irctc1) | Third-party, key required, free tier reportedly very small. Page content could not be fetched; endpoint and response schema **unverified**. | Adapter `rapidapi_irctc` provided, **untested against live responses** (no credential available). |
| railwayapi.com, Apify actors, MCP servers | Third-party / scrapers; terms unverified. | Not implemented. |

## Operating-factor investigation status

| Factor | Evidence available in this project | Label used |
|---|---|---|
| Congestion / headway conflict | Emerges inside the simulation | SIMULATED SCENARIO |
| Station dwell time | Model parameter (assumed 0/1/2 min by class) | SIMULATED SCENARIO |
| Late incoming opposing train | Visible in simulated waits | SIMULATED SCENARIO |
| Maintenance restrictions, weather, infrastructure faults | No provider field or credible live feed integrated | HYPOTHESIS (never attributed) |

## Historical data and provider research (added 2026-10-12)

| Source | Retrieved | Supports |
|---|---|---|
| [HF dataset pjay-in/indian-railways-delays](https://huggingface.co/datasets/pjay-in/indian-railways-delays) | 2026-10-12 | cc-by-4.0. Observed config: NTES-collected per-stop scheduled/actual times (reported or estimated). 2025-02-08..2026-02-07 config: converted from a Kaggle dataset (MIT); actual = scheduled + one delay per stop, so intermediate departures are absent. Used for all learning. |
| [RSTGCN (GitHub)](https://github.com/KoyenaChowdhury/RSTGCN) | 2026-10-12 | Existence of a Sept-2024 delay dataset (no licence stated). Not used. |
| [RailRadar live status docs](https://railradar.in/docs/live-train-status) | 2026-10-12 | Endpoint `GET /v1/trains/{number}/live`, Bearer auth, fields (delayMinutes, currentLocation, route[]), reported free sandbox of 100 requests/month. Adapter written from this; **not run against a live response**. |
| [Railway Engine](https://railengine.vercel.app/) | 2026-10-12 | Another free-tier keyed provider (x-api-key). Not implemented. |
| ntes-client / TrainTrack (unofficial) | 2026-10-12 | Keyless but rely on scraping NTES; may breach NTES terms, not evaluated, not used. |
