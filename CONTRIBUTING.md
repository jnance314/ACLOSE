# Contributing Guidelines

Thank you for your interest in contributing to ACLOSE! We welcome all contributions—whether they’re bug fixes, new features, documentation improvements, or anything else that adds value.

## Getting Started

1. **Fork the Repository**  
   - Click the **Fork** button at the top of this repo to create your own copy.

2. **Clone Your Fork**  
   ```
   git clone https://github.com/your-username/this-repo.git
   ```

3. **Create a Branch**  
   ```
   git checkout -b feature/your-feature-name
   ```

## Development and Testing

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1. **Install Dependencies**  
   ```
   poetry install
   ```

2. **Code Style**  
   We use [Black](https://black.readthedocs.io/) for code formatting. Please ensure your code is formatted before committing:
   ```
   poetry run black .
   ```

3. **Run Tests**  
   We use [pytest](https://docs.pytest.org/en/latest/) (including async support) for our test suite. Make sure all tests pass:
   ```
   poetry run pytest
   ```

## Submitting Your Changes

1. **Commit and Push**  
   ```
   git add .
   git commit -m "Add your descriptive commit message"
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request (PR)**  
   - Go to your fork on GitHub.
   - Click **Compare & pull request**.
   - Provide a clear description of your changes.
   - When you open an issue or pull request, you’ll be guided by one of our templates to help provide all the necessary details.

3. **Ensure Checks Pass**  
   - Our GitHub Actions workflow will automatically run tests and formatting checks.  
   - Please fix any failing checks before requesting a review.

## Reviewing and Merging

- We’ll review your PR for quality, correctness, and adherence to coding standards.  
- If everything looks good, we’ll merge it into the main branch. If changes are needed, we’ll provide feedback.

## Code of Conduct

Please note that this project adheres to a simple standard of respect and inclusivity. By participating, you agree to maintain a welcoming environment for everyone.

## License

This project is licensed under the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.

---


# Project Style Guidelines

Below are some core style guidelines to keep our code consistent and maintainable:

## 1. Type Hints
- Use Python’s built-in type annotations for all function parameters, return types, and class attributes.
- Example:
```
def compute_score(data: list[float]) -> float:
    return sum(data) / len(data)
```

## 2. Docstrings
- Write clear docstrings for modules, classes, and functions.
- Follow a consistent style (e.g., Google-style or reStructuredText).
- Include a description of each parameter, return type, and any exceptions raised.
- Example:
```
def compute_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        point_a (tuple[float, float]): Coordinates of the first point.
        point_b (tuple[float, float]): Coordinates of the second point.

    Returns:
        float: The Euclidean distance between point_a and point_b.
    """
    ...
```

## 3. Comment Style
- Use inline comments (`#`) sparingly for clarifying small logic details.
- For larger explanations, prefer docstrings or dedicated documentation block for readability like:
```
#---
# This is a block comment.
#---
```
- Keep comments relevant and updated if code changes.

## 4. Logging
- Use Python’s built-in `logging` module for runtime information, warnings, and errors instead of print statements.
- Create a named logger (e.g., `logging.getLogger(__name__)` or `logging.getLogger(self.__class__.__name__)`).
- Log messages at the appropriate level (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- Example:
```
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug("This is a debug message.")
logger.info("General info about the process.")
logger.warning("A warning that might need attention.")
```

## 5. Example in Practice
- Here’s a brief example illustrating these guidelines (type hints, docstrings, logging):
```
class ExampleProcessor:
    """
    Processes data according to specified rules.

    Attributes:
        logger (logging.Logger): Logger for this class.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("ExampleProcessor initialized.")

    def process(self, values: list[int]) -> float:
        """
        Process a list of integer values and return their mean.

        Args:
            values (list[int]): A list of integer values.

        Returns:
            float: The mean of the provided values.
        """
        if not values:
            self.logger.warning("Received empty list of values.")
            return 0.0
        return sum(values) / len(values)
```