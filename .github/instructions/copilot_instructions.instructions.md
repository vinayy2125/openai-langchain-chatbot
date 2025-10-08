---
applyTo: '**'
---
1. Preparation Phase:
 
a. PR Validation:
 
PR created properly with clear title and user story tagged.
 
Description should include purpose, changes made, and testing details.
 
 
b. Understand the Context and Purpose of the User Story:
 
Before reviewing code, ensure you understand the objectives and requirements of the PR with acceptance criteria.
 
 
c. Review Approach:
 
Ensure approach aligns with discussions/design shared with senior developers/architects.
 
 
 
---
 
2. Module/Package Structure:
 
a. Modules/packages created at the right place.
b. Directory and package structure should follow project conventions.
c. Proper file naming convention should be used (snake_case for files, PascalCase for classes).
 
 
---
 
3. General Practices / Coding Standards:
 
a. Follow PEP8/PEP20 standards for naming variables, functions, classes, and modules.
b. No circular imports or tight coupling between modules.
c. No unused imports or unused variables.
d. Avoid global variables; prefer function scope or class attributes.
e. Proper formatting in .py files (use Black/Flake8/isort for auto-formatting).
f. Keep __init__.py minimal (only required exports).
g. No hardcoded values; move to config files (.env, yaml, json).
h. Add meaningful docstrings/comments where needed.
i. Type hints (typing) should be used wherever applicable.
 
 
---
 
4. Code Quality / Optimization:
 
a. Refactor complex functions to improve readability and maintainability.
b. Look for opportunities to remove redundant code and unnecessary variables.
c. Add necessary error handling (try-except) for all possible failure cases.
d. Functions/methods should be small, testable, and reusable.
e. No blocking/expensive operations inside synchronous code (use async/multiprocessing where needed).
f. Proper use of Python data structures (dict, set, defaultdict, dataclass).
g. Avoid hardcoded constants; use Enums or constants.py.
h. Follow proper lifecycle/resource management (e.g., close DB connections, files, sessions).
i. Split large scripts into smaller reusable modules.
j. Optimize queries and loops; avoid nested loops when possible.
k. Use logging instead of print statements.
l. Avoid code duplication – move common logic to utility/helper functions.
 
 
---
 
5. Code Reusability:
 
a. Use existing shared utilities/helpers.
b. Avoid duplicate code; create generic, reusable functions or classes.
c. Write code with testability in mind (unit tests, mocks).