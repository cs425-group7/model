"""Naive Bayes diagnosis over Symcat's symptom percentages.

    P(d | evidence) ∝ P(d) · Π_{s present} P(s|d) · Π_{s denied} (1 - P(s|d))

P(s|d) is Symcat's percentage. Symcat lists at most 12 symptoms per condition, so an unlisted symptom gets
a small floor `eps`; percentages are clipped to [eps, 1 - eps] so that one denial cannot zero a condition.
"""
import numpy as np


class SymcatNaiveBayes:
    def __init__(self, conditions: dict, eps: float = 0.01, prior: dict | None = None):
        """conditions: {disease: {symptom: probability}}; prior: {disease: weight}, uniform if None."""
        self.diseases = list(conditions)
        self.symptoms = sorted({s for ps in conditions.values() for s in ps})
        self.index = {s: j for j, s in enumerate(self.symptoms)}
        self.p = np.full((len(self.diseases), len(self.symptoms)), eps)
        for i, d in enumerate(self.diseases):
            for s, prob in conditions[d].items():
                self.p[i, self.index[s]] = min(max(prob, eps), 1 - eps)
        weights = np.array([1.0 if prior is None else prior.get(d, 0.0) for d in self.diseases])
        self.log_prior = np.log(weights / weights.sum())
        self.log_p, self.log_q = np.log(self.p), np.log1p(-self.p)

    def posterior(self, evidence: dict) -> np.ndarray:
        """evidence: {symptom: True/False}; symptoms outside the model are ignored."""
        logits = self.log_prior.copy()
        for s, present in evidence.items():
            j = self.index.get(s)
            if j is not None:
                logits += self.log_p[:, j] if present else self.log_q[:, j]
        logits -= logits.max()
        post = np.exp(logits)
        return post / post.sum()

    def rank(self, evidence: dict, k: int | None = None) -> list:
        post = self.posterior(evidence)
        order = np.argsort(-post)[:k]
        return [(self.diseases[i], float(post[i])) for i in order]

    def information_gain(self, evidence: dict) -> dict:
        """Expected entropy reduction (bits) from asking about each symptom not yet in the evidence."""
        post = self.posterior(evidence)

        def entropy(x):
            x = x / x.sum(axis=0, keepdims=True)
            return -(x * np.log2(np.clip(x, 1e-12, None))).sum(axis=0)

        p_yes = post @ self.p                                    # P(answer yes) per symptom
        h_yes = entropy(post[:, None] * self.p)
        h_no = entropy(post[:, None] * (1 - self.p))
        h_now = -(post * np.log2(np.clip(post, 1e-12, None))).sum()
        gain = h_now - (p_yes * h_yes + (1 - p_yes) * h_no)
        return {s: float(gain[j]) for s, j in self.index.items() if s not in evidence}
