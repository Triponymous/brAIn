# Contributing to brAIn

Welcome to the brAIn project! We're building a neuromorphic AI companion by combining spiking neural networks with LLM speech capabilities. This is active research, and we'd love your contributions.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Rust toolchain (for pet-face app)
- Git

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Triponymous/brAIn.git
   cd brAIn
   ```

2. **Python setup** (using `uv`)
   ```bash
   uv venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   uv sync
   ```

3. **Frontend setup** (Node.js)
   ```bash
   cd ui
   npm install
   cd ..
   ```

4. **Rust setup** (pet-face app)
   ```bash
   cd pet-face
   rustup update
   cargo build
   cd ..
   ```

## Code Style

### Python
We use `ruff` for linting and formatting:
```bash
ruff check .
ruff format .
```

### TypeScript/JavaScript
We use ESLint and Prettier:
```bash
cd ui
npm run lint
npm run format
cd ..
```

## Testing

Run pytest with verbose output:
```bash
.venv/bin/pytest -v
```

Tests should cover new functionality and pass before submitting a PR.

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Keep PRs focused**: One feature or fix per PR. Smaller PRs are easier to review.

3. **Describe your changes**: Clearly explain what you changed and why. Reference any related issues.

4. **Pass tests and style checks**: Ensure all tests pass and code meets style standards before submitting.

5. **Submit and engage**: We'll review and provide feedback. Be responsive to questions.

## Areas Where We Need Help

- **ESP32 firmware**: Sensor integration and wireless communication
- **Sensor adapters**: New hardware drivers and data collection modules
- **Dashboard components**: Visualization and UI improvements
- **Documentation**: API docs, tutorials, architecture guides

## Code of Conduct

Be respectful and constructive. We're a collaborative research community. Disagreements are welcome; personal attacks are not.

## Communication

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas
- **PRs**: The primary way to propose changes

---

Thank you for contributing to brAIn. Your work helps advance neuromorphic AI research.

*Project created by Leon Matthies*
