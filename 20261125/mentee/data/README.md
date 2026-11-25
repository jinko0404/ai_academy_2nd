# Day28 평가 데이터 계약

- `corpus.json`: Day27에서 이어받은 합성 업무 청크 24개 (현재/ready/권한 데이터 구분)
- `questions.json`: Target 입력용 질문 8개. **Gold chunk ID는 여기 없습니다.** `reference_answer`는 평가자 참고용이므로 Target에 전달하지 않습니다.
- `gold_labels.json`: 질문별 Gold·graded relevance·버전·검토 정책. 0등급도 실제 판정되었다고 명시합니다.
- `fixture_runs.json`: 4개 변형 × 8개 질문의 **고정 교육용 순위/답변/시간**, 실측 결과가 아닙니다.
- `failure_cases.json`: 검색/순위/생성/시스템 장애 진단 예시
- `generation_claims.json`: Claim 지원 여부에 대한 교육용 사람 레이블
- `domain_examples/`: 금융 정형 CSV+비정형 FAQ, 의료행정 정형 CSV+비정형 FAQ

**중요:** Gold는 Target Runner 입력에 주지 않습니다. Evaluation에서만 사용합니다. 실측·고정 데이터를 한 보고서에서 평균내지 않습니다. No-answer는 Retrieval 평균 제외 후 별도 집계합니다.
