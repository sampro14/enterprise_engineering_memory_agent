from eval.metrics import grade
from eval.scenarios import Question, generate


def q(kind="current", expected=("Razorpay",), forbidden=("Stripe",)):
    return Question("q", "s", "change", kind, "?", list(expected), list(forbidden))


def test_grading():
    assert grade(q(), ["Razorpay"], False).correct
    g = grade(q(), ["Razorpay", "Stripe"], False)
    assert not g.correct and g.stale
    assert not grade(q(), [], True).correct
    assert grade(q("when", ["2026-09-11"], []), ["2026-09-11"], False).correct
    assert grade(q("multi", ["Redis", "Kafka"], []), ["redis", "Kafka", "S3"], False).correct
    assert not grade(q("multi", ["Redis", "Kafka"], []), ["Redis"], False).correct


def test_world_is_deterministic_and_well_formed():
    a, b = generate(3), generate(3)
    assert [d.text for d in a.docs] == [d.text for d in b.docs]
    assert generate(4).docs[0].text != a.docs[0].text or generate(4).docs[1].text != a.docs[1].text
    assert all(x.ingest_date <= a.asked_at for x in a.docs)
    assert all(qq.expected for qq in a.questions)
    dates = [d.ingest_date for d in a.docs]
    assert dates == sorted(dates)
    ids = [qq.id for qq in a.questions]
    assert len(ids) == len(set(ids))
    # each stale/forbidden value must differ from its expected value
    for qq in a.questions:
        assert not set(map(str.lower, qq.expected)) & set(map(str.lower, qq.forbidden))
