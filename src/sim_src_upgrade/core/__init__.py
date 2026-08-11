"""고속 sim 코어 (src/sim_src 사본, 상대 import).

모듈 구성은 원본과 1:1 대응한다:
    EntityManager / EventManager / ScenarioManager / MCIEnvironment_gymnasium
    RuleManager / ShinHeuristics / ShinAlignedHeuristics

여기서는 재수출을 하지 않는다 — 사용측이 필요한 모듈만 import 해야
pandas 등 무거운 의존이 불필요하게 끌려오지 않는다.
"""
