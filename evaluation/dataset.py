"""Hardcoded evaluation dataset for benchmarking CRAG vs naive RAG."""

from __future__ import annotations

EVAL_DATASET: list[dict[str, str]] = [
    {
        "question": "What is the company's policy on remote work and how many days per week can employees work from home?",
        "ground_truth": "Employees can work remotely up to 3 days per week. They must be in-office on Tuesdays and Thursdays for team collaboration. Remote work requests beyond the standard policy require manager and HR approval.",
    },
    {
        "question": "How does the onboarding process work for new engineering hires?",
        "ground_truth": "New engineering hires go through a 2-week structured onboarding: Week 1 covers IT setup, security training, codebase walkthroughs, and buddy pairing. Week 2 focuses on first PR submission, architecture deep-dives, and meeting stakeholders.",
    },
    {
        "question": "What is the technical architecture of the main product's backend?",
        "ground_truth": "The backend uses a microservices architecture running on Kubernetes. Core services are written in Python (FastAPI) and Go. Data is stored in PostgreSQL for transactional data and Redis for caching. Inter-service communication uses gRPC with an API gateway handling external REST traffic.",
    },
    {
        "question": "What are the annual leave and PTO policies?",
        "ground_truth": "Full-time employees receive 25 days of PTO per year, accrued monthly. Unused PTO can carry over up to 5 days into the next year. Sick leave is separate and unlimited with a doctor's note required after 3 consecutive days.",
    },
    {
        "question": "How does the company handle data privacy and GDPR compliance?",
        "ground_truth": "The company maintains GDPR compliance through a dedicated DPO, annual privacy impact assessments, data encryption at rest and in transit, 30-day data deletion requests, and mandatory privacy training for all employees completed annually.",
    },
    {
        "question": "What CI/CD pipeline tools and practices does the engineering team use?",
        "ground_truth": "The team uses GitHub Actions for CI with automated linting, unit tests, and integration tests on every PR. CD is handled by ArgoCD deploying to Kubernetes clusters. Feature flags are managed through LaunchDarkly for progressive rollouts.",
    },
    {
        "question": "What is the current market capitalization of Tesla?",
        "ground_truth": "This question is outside the scope of internal company documentation. The system should recognise this as out-of-domain and either perform a web search or state it cannot answer from the available knowledge base.",
    },
    {
        "question": "Describe the incident response procedure when a production outage occurs.",
        "ground_truth": "On detecting a P1 outage, the on-call engineer is paged via PagerDuty within 5 minutes. A war room is opened in Slack, an incident commander is assigned, and status updates go out every 30 minutes. Post-incident, a blameless RCA is required within 48 hours and tracked in the incident database.",
    },
    {
        "question": "What programming languages does the company officially support?",
        "ground_truth": "The company officially supports Python, Go, and TypeScript. Python is used for ML/data pipelines and internal tools, Go for performance-critical microservices, and TypeScript for all frontend and BFF (Backend-for-Frontend) layers. Other languages require architecture review approval.",
    },
    {
        "question": "Who won the FIFA World Cup in 2022 and what was the final score?",
        "ground_truth": "This question is not related to internal company documentation. It should trigger web search or the system should state the information is not available in the knowledge base. Argentina won on penalties against France (3-3 after extra time).",
    },
]
