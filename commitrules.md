## Commit Guidelines

### ## Commit Message Structure

All commits must follow this format:
```
<type>: <description>

<body>

<footer>
```

### ## Commit Types

Use these human-friendly commit types:

- **fix**: Bug fixes and corrections
- **add**: New features or functionality
- **update**: Modifications to existing features
- **remove**: Deletion of code or features
- **refactor**: Code restructuring without functional changes
- **docs**: Documentation changes
- **style**: Formatting and code style changes
- **test**: Adding or updating tests

### ## Title Guidelines

#### ## Writing Style
- Write in present tense, imperative mood
- Keep under 50 characters
- No period at the end
- Be specific and descriptive
- Avoid generic terms like "feat", "task", "implement"

#### ## Good Examples
```
fix: resolve login timeout issue
add: user profile photo upload
update: improve search performance
remove: deprecated payment methods
```

#### ## Avoid These Patterns
```
❌ feat: add new feature
❌ task: implement user authentication  
❌ update: various improvements
❌ fix: bug fixes
```

### ## Commit Message Body

#### ## Content Requirements
- Explain **what** and **why**, not how
- Use bullet points for multiple changes
- Reference issue numbers when applicable
- Keep lines under 72 characters
- Leave blank line between title and body

#### ## Example Format
```
fix: resolve memory leak in image processing

- Clear image cache after processing completion
- Add proper cleanup in error handling paths
- Prevents application crashes during batch operations

Closes #234
```

### ## Authorship Requirements

#### ## Author Information
- All commits must be authored by you personally
- Use consistent name and email across all commits
- No co-authored-by tags with AI assistants or agents
- Maintain single author attribution

#### ## Git Configuration
```bash
git config user.name "Your Name"
git config user.email "your.email@domain.com"
```

### ## Code Review Template

#### ## For Bug Fixes
```markdown
## Problem
Brief description of the issue being resolved

## Solution
Explanation of the fix implemented

## Testing
- [ ] Manual testing completed
- [ ] Edge cases verified
- [ ] No regression introduced

## Impact
Areas of the codebase affected
```

#### ## For New Features
```markdown
## Overview
What functionality is being added

## Implementation
Key technical decisions and approach

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests passing
- [ ] User acceptance criteria met

## Documentation
- [ ] Code comments added
- [ ] README updated if needed
- [ ] API docs updated if applicable
```

#### ## For Refactoring
```markdown
## Motivation
Why this refactoring is necessary

## Changes
- List of structural changes made
- Performance improvements expected

## Verification
- [ ] Functionality unchanged
- [ ] Tests still passing
- [ ] No breaking changes introduced
```

### ## Best Practices

#### ## Commit Frequency
- Make small, focused commits
- Each commit should represent one logical change
- Commit working code frequently
- Avoid massive commits with multiple unrelated changes

#### ## Quality Standards
- Test your changes before committing
- Ensure code follows project style guidelines
- Write clear, self-documenting code
- Include relevant comments for complex logic

#### ## Review Process
- Self-review your changes before submitting
- Provide context in PR descriptions
- Respond promptly to review feedback
- Keep discussions focused and constructive

### ## Common Mistakes to Avoid

- Vague commit messages like "updates" or "changes"
- Mixing multiple unrelated changes in one commit
- Committing broken or untested code
- Using AI-generated commit messages without personalization
- Forgetting to reference related issues or tickets
- Inconsistent formatting across commit messages