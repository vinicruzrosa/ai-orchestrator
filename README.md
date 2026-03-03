# 🦉 FluentBase: AI-Powered English Tutor

**FluentBase** is an AI orchestrator built with **FastAPI** and **Groq Cloud**. It acts as a personalized English tutor by analyzing sentences, providing pedagogical corrections, and suggesting natural alternatives.

---

## 🏗️ Project Architecture

The project follows **Hexagonal Architecture** (Ports & Adapters) to ensure business logic remains independent of external tools.

* **`/app/domain`**: Business entities and core exceptions.
* **`/app/application/ports`**: Interfaces (Abstract Base Classes) defining service contracts.
* **`/app/adapters`**: Technical implementations (Groq API, DTOs, Handlers).
* **`/tests`**: Suite of unit and integration tests.

---

## 🚀 Quick Start

### 1. Environment Setup
Create a `.env` file or export the variable in your terminal:
```bash
# Set your Groq API Key
export FLUENT_BASE_API_KEY="your_api_key_here"