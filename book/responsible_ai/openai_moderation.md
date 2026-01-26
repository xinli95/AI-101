# OpenAI Moderation API

The OpenAI Moderation API helps you identify **potentially harmful** content in **text and images**, so you can filter, block, or route content for review before it reaches users. It returns category flags and scores you can use to build policy‑aware safety checks.

## 1. What moderation does

- **Classifies inputs** (text and/or images) against a safety taxonomy.
- **Returns flags and scores** per category so you can automate actions (block, allow, review).
- **Supports multimodal inputs** with `omni-moderation-latest` and a legacy text‑only model for backward compatibility.

## 2. When to use it

Use moderation as a front‑line safety check for:

- user‑generated content before display
- user prompts before model inference
- model outputs before delivery

A common pattern is **pre‑filter → model → post‑filter**, with moderation on both the input and the output for higher coverage.

## 3. Models

- **`omni-moderation-latest`**: recommended; supports broader categories and multimodal inputs.
- **`text-moderation-latest`**: legacy; text‑only with fewer categories.

## 4. Output fields (what you get back)

The moderation response includes:

- **`flagged`**: overall indicator of potentially harmful content
- **`categories`**: per‑category boolean flags
- **`category_scores`**: per‑category confidence scores (0–1)
- **`category_applied_input_types`**: which input types (text/image) triggered the category (omni models only)

Note: if your logic depends on **`category_scores`**, re‑calibration may be needed as the underlying model improves over time.

## 5. Content classification taxonomy

Below is the current moderation taxonomy, copied from the moderation guide. Categories marked **Text only** do not support image‑only inputs (image‑only requests return 0 for those categories).

| Category | Description | Inputs |
| --- | --- | --- |
| `harassment` | Content that expresses, incites, or promotes harassing language towards any target. | Text only |
| `harassment/threatening` | Harassment content that also includes violence or serious harm towards any target. | Text only |
| `hate` | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non‑protected groups (e.g., chess players) is harassment. | Text only |
| `hate/threatening` | Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. | Text only |
| `illicit` | Content that gives advice or instruction on how to commit illicit acts (e.g., “how to shoplift”). | Text only |
| `illicit/violent` | Content like `illicit` that also includes references to violence or procuring a weapon. | Text only |
| `self-harm` | Content that promotes, encourages, or depicts acts of self‑harm, such as suicide, cutting, and eating disorders. | Text and images |
| `self-harm/intent` | Content where the speaker expresses that they are engaging or intend to engage in acts of self‑harm, such as suicide, cutting, and eating disorders. | Text and images |
| `self-harm/instructions` | Content that encourages performing acts of self‑harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts. | Text and images |
| `sexual` | Content meant to arouse sexual excitement, such as descriptions of sexual activity, or that promotes sexual services (excluding sex education and wellness). | Text and images |
| `sexual/minors` | Sexual content that includes an individual who is under 18 years old. | Text only |
| `violence` | Content that depicts death, violence, or physical injury. | Text and images |
| `violence/graphic` | Content that depicts death, violence, or physical injury in graphic detail. | Text and images |

## Sources

- [OpenAI Moderation guide](https://platform.openai.com/docs/guides/moderation#content-classifications)
