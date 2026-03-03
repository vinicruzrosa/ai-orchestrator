from app.domain.entities import ChatAnalysis

def test_chat_analysis_entity():
    analysis = ChatAnalysis(
        message="I has a car",
        is_correct=False,
        explanation="Verb agreement error.",
        reply="What color is it?",
        inferred_context="Casual",
        correction="I have a car",
        suggestions=["I've got a car"]
    )
    assert analysis.is_correct is False
    assert analysis.correction == "I have a car"
    assert "Casual" in analysis.inferred_context