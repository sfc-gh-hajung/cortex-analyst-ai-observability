# Contributing

Thank you for your interest in contributing to Cortex Analyst AI Observability!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cortex-analyst-ai-observability.git
   ```
3. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your credentials
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

## Testing

Before submitting a PR:

1. Test the main evaluation script:
   ```bash
   python cortex_analyst_observability.py
   ```

2. Test the Streamlit app:
   ```bash
   streamlit run analyst_evalset_generator.py
   ```

3. Verify results appear in Snowsight under AI & ML → Evaluations

## Submitting Changes

1. Commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a Pull Request with:
   - Clear description of changes
   - Any relevant issue numbers
   - Screenshots if UI changes

## Reporting Issues

When opening an issue, include:

- Python version
- Snowflake account region
- Error messages (sanitize credentials)
- Steps to reproduce

## Questions?

Open an issue with the `question` label.
