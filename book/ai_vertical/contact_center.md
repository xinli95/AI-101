# Contact Center: From Automation to Customer Experience

## Insights from Ping Wu

In Sequoia's [*Why AI Will Create Abundance and Transform Customer Experience*](https://sequoiacap.com/podcast/why-ai-will-create-abundance-and-transform-customer-experience-cresta-ceo-ping-wu), Cresta CEO Ping Wu connects customer experience with practical deployment. The following summarizes his views at the time of the interview:

1. **Prevent avoidable contacts.** Analyze why customers seek help and repair underlying product or process failures.
2. **Combine automation with assistance.** Automate suitable conversations; support humans with retrieval, data entry, and after-call work elsewhere.
3. **Learn from real work.** Observe conversations and screen activity to understand workflows and build realistic simulations.
4. **Build the complete system.** Legacy interfaces, transfers, long calls, privacy, and data residency make production harder than a demo.
5. **Match methods to tasks.** Cresta combines models, prompting, retrieval, fine-tuning, and distillation. RL for end-to-end outcomes remained exploratory.
6. **Expand what service can offer.** Multilingual and asynchronous interactions can unlock previously unaffordable experiences. Consumer assistants may also initiate contacts.
7. **Maintain a continuous relationship.** Connect sales, support, and feedback through shared context across the customer journey.

**Attribution note:** The arguments about adoption speed and value accruing toward applications were explicitly articulated by co-guest Doug Leone.

## Applying the ideas: diagnose the gap before choosing a method

The example below is this book's analysis, not a description of Cresta's implementation. Consider an assistant handling a delivery problem. Its success criterion is that the customer receives an accurate explanation and the appropriate action is completed.

| Observed failure | Question to investigate | Evidence to collect |
| --- | --- | --- |
| Gives an outdated delivery estimate | Is the current order record available to the assistant? | Whether retrieved context matches the order system |
| Promises a replacement that never ships | Can it execute and confirm the replacement action? | Tool result and resulting order state |
| Omits details when escalating | Does it know the required handoff structure? | Completeness of the receiving agent's case record |
| Repeats a replacement after a retry | Does the surrounding application prevent duplicate actions? | Behavior when the same request is retried |
| Ends the conversation quickly but leaves the issue unresolved | Does the evaluation measure completion adequately? | Resolution, repeat contact, and customer feedback |

These failures require different interventions. Retrieval can supply missing records; tool integration enables actions; demonstrations may help teach a handoff format; application logic must handle retries. The [post-training method-selection framework](../post_training/overview.md) helps identify where training fits.

## Questions for evaluating a contact center application

These are proposed review questions for this case study:

- Can the assistant distinguish a missing fact from an action it cannot perform?
- When it hands off to a person, does the customer need to explain the problem again?
- Does a claimed resolution match the state of the relevant business system?
- Are improvements preserved across languages and uncommon cases?
- What happens to resolution quality when latency or serving cost is reduced?

Use representative held-out cases to assess these questions alongside cost and response time.
