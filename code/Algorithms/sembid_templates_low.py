"""
Low-conversion SemBid templates (random sampling).
"""
import numpy as np
import random


class BiddingLanguageGeneratorWithTask:
    """Generate SemBid guidance (History + Strategy + Task) for low-conv."""

    def __init__(
        self,
        roi_thresholds=(0.1, 5.0, 40.0),
        cvr_change_thresholds=(0.0, 0.0),
        cpa_change_thresholds=(500, 2000),
        pvalue_thresholds=(0.07, 0.11),
        budget_ratio_thresholds=(0.001, 0.004),
    ):
        """
        Args:
            roi_thresholds: ROI thresholds (low, moderate, good)
            cvr_change_thresholds: CVR thresholds (small, large) [kept for parity]
            cpa_change_thresholds: CPA change thresholds (small, large)
            pvalue_thresholds: pValue thresholds (low, high)
            budget_ratio_thresholds: remaining budget ratio thresholds (low, high)
        """
        self.roi_low, self.roi_moderate, self.roi_good = roi_thresholds
        self.cvr_small, self.cvr_large = cvr_change_thresholds
        self.cpa_small, self.cpa_large = cpa_change_thresholds
        self.pvalue_low, self.pvalue_high = pvalue_thresholds
        self.budget_low, self.budget_high = budget_ratio_thresholds

        self.task_templates = [
            "Balance conversion volume and cost efficiency with target CPA {cpa:.1f}.",
            "Maximize conversions while maintaining CPA below {cpa:.1f}.",
            "Optimize bidding to achieve target CPA of {cpa:.1f}.",
            "Control cost per acquisition to stay within {cpa:.1f}.",
        ]

        self.history_templates = {
            "roi_low": [
                "The ROI was low after the last bid.",
                "Previous bid resulted in low return on investment.",
                "Last action led to poor ROI performance.",
                "The bid was not profitable.",
                "ROI dropped below expectations.",
            ],
            "roi_moderate": [
                "The ROI was moderate after the last bid.",
                "Previous bid achieved acceptable ROI.",
                "Last action resulted in moderate returns.",
                "The bid showed moderate profitability.",
                "ROI was within acceptable range.",
            ],
            "roi_good": [
                "The ROI was good after the last bid.",
                "Previous bid achieved high return on investment.",
                "Last action resulted in excellent ROI.",
                "The bid was highly profitable.",
                "ROI exceeded expectations.",
            ],
            "cvr_increase": [
                "A conversion happened after the bid.",
                "Conversion was observed following the action.",
                "The bid led to a conversion.",
                "Conversion occurred after this bid.",
                "A conversion was achieved.",
            ],
            "cvr_none": [
                "No conversion was observed after the bid.",
                "No conversion happened following the action.",
                "The bid did not lead to conversion.",
                "No conversion was recorded.",
                "Conversion did not occur this time.",
            ],
            "cpa_increase": [
                "Cost per acquisition increased.",
                "CPA rose after the bid.",
                "Acquisition cost went up.",
                "CPA showed upward trend.",
                "Cost efficiency decreased.",
            ],
            "cpa_decrease": [
                "Cost per acquisition decreased.",
                "CPA dropped after the bid.",
                "Acquisition cost went down.",
                "CPA showed downward trend.",
                "Cost efficiency improved.",
            ],
        }

        self.strategy_templates = {
            "conservative": [
                "Consider bidding conservatively to maintain budget.",
                "Use a conservative bid to preserve resources.",
                "A moderate bid would be safer now.",
                "Keep the bid conservative to avoid overspending.",
                "Maintain conservative bidding strategy.",
            ],
            "moderate": [
                "A moderate bid would be appropriate.",
                "Consider a balanced bidding approach.",
                "Use a moderate bid to balance risk and reward.",
                "Moderate bid would help maintain stability.",
                "Try a moderate bidding strategy.",
            ],
            "aggressive": [
                "Consider increasing the bid to capture more opportunities.",
                "An aggressive bid might capture better conversions.",
                "Raise the bid to compete more effectively.",
                "Higher bid could improve win rate.",
                "Consider bidding more aggressively.",
            ],
            "high_pvalue": [
                "High pValue suggests good conversion potential. Consider higher bid.",
                "The pValue is favorable, you can bid higher.",
                "Strong pValue indicates good opportunity.",
                "High pValue means better conversion probability.",
                "Favorable pValue supports higher bidding.",
            ],
            "low_pvalue": [
                "Low pValue suggests lower conversion potential. Consider conservative bid.",
                "The pValue is low, bid conservatively.",
                "Weak pValue indicates limited opportunity.",
                "Low pValue means lower conversion probability.",
                "Unfavorable pValue suggests lower bidding.",
            ],
            "budget_low": [
                "Remaining budget is low. Bid conservatively.",
                "Limited budget remaining, use cautious bidding.",
                "Low budget requires careful bid management.",
                "Conserve budget with moderate bids.",
                "Budget constraint suggests conservative approach.",
            ],
            "budget_high": [
                "Remaining budget is sufficient. You can bid more aggressively.",
                "Adequate budget allows for higher bids.",
                "Good budget availability supports competitive bidding.",
                "Sufficient budget enables flexible bidding strategy.",
                "Budget is healthy, consider more aggressive bids.",
            ],
        }

    def _extract_features(self, state):
        if isinstance(state, np.ndarray):
            state = state.flatten()
        else:
            state = np.array(state).flatten()

        if len(state) >= 16:
            return {
                "budget": state[3] if len(state) > 3 else 0,
                "remaining_budget": state[6] if len(state) > 6 else 0,
                "pvalue": state[8] if len(state) > 8 else 0,
                "cost": state[13] if len(state) > 13 else 0,
                "conversion": state[15] if len(state) > 15 else 0,
            }
        return {
            "budget": 0,
            "remaining_budget": 0,
            "pvalue": 0,
            "cost": 0,
            "conversion": 0,
        }

    def _calculate_roi(self, prev_state, reward, current_state):
        prev_features = self._extract_features(prev_state)
        curr_features = self._extract_features(current_state)

        cost = curr_features.get("cost", 0) - prev_features.get("cost", 0)
        conversion = curr_features.get("conversion", 0) - prev_features.get("conversion", 0)

        if cost > 0:
            roi = (conversion * 10.0 - cost) / cost
        else:
            roi = 0
        return roi

    def generate_task_description(self, target_cpa):
        return random.choice(self.task_templates).format(cpa=target_cpa)

    def generate_history(self, prev_state, prev_action, reward, current_state):
        if prev_state is None or current_state is None:
            return ""

        texts = []
        roi = self._calculate_roi(prev_state, reward, current_state)
        if roi < self.roi_low:
            texts.append(random.choice(self.history_templates["roi_low"]))
        elif roi < self.roi_good:
            texts.append(random.choice(self.history_templates["roi_moderate"]))
        else:
            texts.append(random.choice(self.history_templates["roi_good"]))

        if reward > 0:
            texts.append(random.choice(self.history_templates["cvr_increase"]))
        else:
            texts.append(random.choice(self.history_templates["cvr_none"]))

        prev_features = self._extract_features(prev_state)
        curr_features = self._extract_features(current_state)
        cost_change = curr_features.get("cost", 0) - prev_features.get("cost", 0)

        if cost_change > self.cpa_large:
            texts.append(random.choice(self.history_templates["cpa_increase"]))
        elif cost_change < -self.cpa_small and cost_change < 0:
            texts.append(random.choice(self.history_templates["cpa_decrease"]))

        return " ".join(texts) if texts else "Previous bid was executed."

    def generate_strategy(self, current_state, suggested_bid):
        if current_state is None:
            return "Start bidding conservatively."

        features = self._extract_features(current_state)
        texts = []

        pvalue = features.get("pvalue", 0)
        if pvalue > self.pvalue_high:
            texts.append(random.choice(self.strategy_templates["high_pvalue"]))
        elif pvalue < self.pvalue_low:
            texts.append(random.choice(self.strategy_templates["low_pvalue"]))

        remaining_budget = features.get("remaining_budget", 0)
        budget = features.get("budget", 1)
        if budget > 0:
            budget_ratio = remaining_budget / budget
            budget_ratio = min(max(budget_ratio, 0.0), 1.0)
        else:
            budget_ratio = 0.0

        if budget_ratio < self.budget_low:
            texts.append(random.choice(self.strategy_templates["budget_low"]))
        elif budget_ratio > self.budget_high:
            texts.append(random.choice(self.strategy_templates["budget_high"]))

        if suggested_bid < 10:
            texts.append(random.choice(self.strategy_templates["conservative"]))
        elif suggested_bid > 50:
            texts.append(random.choice(self.strategy_templates["aggressive"]))
        else:
            texts.append(random.choice(self.strategy_templates["moderate"]))

        return " ".join(texts) if texts else f"Consider bidding around {suggested_bid:.2f}."

    def generate_all(self, prev_state, prev_action, reward, current_state, suggested_bid, target_cpa):
        task = self.generate_task_description(target_cpa)
        history = self.generate_history(prev_state, prev_action, reward, current_state)
        strategy = self.generate_strategy(current_state, suggested_bid)
        return task, history, strategy
