# Issues Found in LaTeX Files

This document lists all TODOs found in the .tex files of this project. These can be used to create GitHub issues for tracking and resolution.

## Issue 1: Add Geometric Explanation for Separable Case

**File:** `report/mathematical-background.tex` (Line 82)

**Context:** Section on Solvability, discussing the separable case in logistic regression

**Description:**
Add a geometric explanation of what is happening in the separable case. Currently, the mathematical explanation shows that when data is linearly separable, the optimization problem has no finite solution because weights can be scaled arbitrarily large. A geometric interpretation would help readers understand this phenomenon visually.

**Current TODO:**
```latex
\TODO{Explain what is happening in the seperable case from a geometric point of view!}
```

**Location in document:** After the mathematical proof showing that the minimum is not attained at a finite parameter when data is separable.

**Suggested approach:**
- Explain how the decision boundary moves as parameters scale
- Describe how the margin between classes increases without bound
- Possibly add a figure or diagram showing this geometric behavior

---

## Issue 2: Explain Benefits of Regularization

**File:** `report/mathematical-background.tex` (Line 88)

**Context:** Subsection on Regularization

**Description:**
The regularization subsection currently only states "solves some problems" but lacks a detailed explanation of why regularization is beneficial. This section should explain:
- How regularization prevents weights from growing unbounded
- Why this helps with the separability problem discussed earlier
- Other benefits of regularization (generalization, preventing overfitting)

**Current TODO:**
```latex
solves some problems \TODO{explain why regularization is a good thing to do!}
```

**Location in document:** Start of the Regularization subsection (section 1.3)

**Suggested approach:**
- Connect back to the separability problem
- Explain how regularization ensures unique, finite solutions
- Discuss improved generalization performance
- Mention common regularization techniques (L1, L2)

---

## Issue 3: Document Smoothness Property Placement

**File:** `report/mathematical-background.tex` (Line 90)

**Context:** End of Regularization subsection

**Description:**
It is noted that the optimization problem is C∞ (infinitely differentiable), but there's uncertainty about where this property should be documented in the report. This is an important property for discussing optimization algorithms like gradient descent.

**Current TODO:**
```latex
\TODO{It is fairly obvious that the optimization problem is $C^\infty$. Where should we put this?}
```

**Location in document:** End of the Regularization subsection

**Suggested approach:**
- Determine the most logical section for discussing smoothness properties
- Options include:
  - In the Solvability subsection where convexity is discussed
  - In a new subsection on properties of the optimization problem
  - At the beginning of the Stochastic Gradient Descent section (if it relies on smoothness)
- Add a brief proof or reference for the C∞ property
- Explain why this property is important for the chosen optimization method

---

## Summary

Total TODOs found: **3**

All TODOs are located in: `report/mathematical-background.tex`

These issues primarily concern expanding the mathematical explanations in the report to make it more complete and understandable for readers.
