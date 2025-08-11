# behavior_model.py

def get_behavioral_score(
    research_depth_value: int,
    num_unique_sources: int,
    decision_speed_category: str,
    logins_per_month: int,
    panic_sells_after_bad_news: bool,
    buy_the_dip_after_downturn: bool,
    strategic_rebalancing_after_news: bool,
    transaction_reversals_per_month: int,
    panic_selling_incidents: int,
    financial_literacy_score: int,
    volatility_level: int,
    amount_invested: float
) -> float:
    behavioral_risk_score = 0

    # Research Depth
    raw_depth_risk = (5 - research_depth_value) * 2
    if financial_literacy_score > 15:
        research_depth_risk_contribution = raw_depth_risk * 0.8
    elif financial_literacy_score < 5:
        research_depth_risk_contribution = raw_depth_risk * 1.2
    else:
        research_depth_risk_contribution = raw_depth_risk
    behavioral_risk_score += research_depth_risk_contribution

    # Source Diversity
    raw_diversity_risk = 0
    if 1 <= num_unique_sources <= 2:
        raw_diversity_risk = +5
    elif num_unique_sources >= 6:
        raw_diversity_risk = -5
    if financial_literacy_score > 15:
        source_diversity_risk_contribution = raw_diversity_risk * 0.8
    elif financial_literacy_score < 5:
        source_diversity_risk_contribution = raw_diversity_risk * 1.2
    else:
        source_diversity_risk_contribution = raw_diversity_risk
    behavioral_risk_score += source_diversity_risk_contribution

    # Decision Speed
    base_speed_risk_points = 0
    if decision_speed_category == "immediate":
        base_speed_risk_points = 7
    elif decision_speed_category == "very slow":
        base_speed_risk_points = 5
    final_speed_risk_points = base_speed_risk_points
    if base_speed_risk_points > 0:
        if volatility_level > 3 and amount_invested > 100000:
            final_speed_risk_points *= 1.3
        elif volatility_level < 2 and amount_invested < 10000:
            final_speed_risk_points *= 0.7
    elif base_speed_risk_points == 0:
        if volatility_level <= 1 and amount_invested <= 5000:
            final_speed_risk_points = 1
    behavioral_risk_score += final_speed_risk_points

    # Monitoring Frequency
    if 1 <= logins_per_month <= 3:
        behavioral_risk_score += 5
    elif logins_per_month >= 11:
        behavioral_risk_score += 8

    # News Response
    news_response_risk = 0
    if panic_sells_after_bad_news:
        news_response_risk += 10
    if buy_the_dip_after_downturn:
        news_response_risk -= 5
    if strategic_rebalancing_after_news:
        news_response_risk -= 3
    behavioral_risk_score += news_response_risk

    # Decision Reversals
    if 2 <= transaction_reversals_per_month <= 4:
        behavioral_risk_score += 5
    elif transaction_reversals_per_month >= 5:
        behavioral_risk_score += 10

    # Panic Selling Incidents
    if panic_selling_incidents == 1:
        behavioral_risk_score += 15
    elif panic_selling_incidents >= 2:
        behavioral_risk_score += 25

    return max(0, round(behavioral_risk_score, 2))
