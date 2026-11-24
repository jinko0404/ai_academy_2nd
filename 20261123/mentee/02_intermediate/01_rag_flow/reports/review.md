# 실습 01 확인 기록

- 선택한 수준: 02_intermediate
- 실행일: 2026. 11. 23
- 실행 명령: python main.py
- 결과 파일: outputs/rag_flow.json
- 입력 출처와 실행한 구성요소: 멘티 폴더의 `data/lecture_hits.json`에서 강의 예시 문서를 읽었다. 실행 모드는 `concept_review`이며, 로컬 Python 코드가 예시 문서를 선택하고 단계 목록과 기대 판단을 JSON으로 저장했다. 결과의 `execution.db`, `execution.embedding`, `execution.llm`은 모두 `false`였다.

## 1. 검색 결과가 있는데도 insufficient_context인 이유는 무엇인가요?

`outputs/rag_flow.json`의 `case.document_found`는 `true`다. 과제 문서(`case.doc_id` = `DOC-ASSIGN`)를 찾았기 때문이다. 하지만 `case.evidence`에는 정확한 마감 날짜가 없어 `case.exact_date_available`은 `false`다. 현재 근거로는 “내 과제의 정확한 마감 날짜는?”에 답할 수 없으므로 `case.expected_status`를 `insufficient_context`로 기록했다.

## 2. 원문의 어떤 표현을 근거로 판단했나요?

`case.evidence`의 “제출 기한은 담당자가 공지한 날짜를 따릅니다.”를 근거로 판단했다. 이 문장은 제출 기한을 어디서 확인해야 하는지 알려 주며, 그 날짜를 확인하려면 담당자의 공지가 필요하다. 이 문장의 출처는 `case.citation_id` 202, `case.source_version` `v1`, `case.source_locator` `assignment.txt`로 확인했다.

## 3. find_hit 함수가 의미 검색을 수행하나요?

`find_hit()`은 `load_hits()`가 읽은 예시 목록을 순서대로 확인하면서 `hit["doc_id"] == doc_id`를 비교한다. 이 실행에서는 `DOC-ASSIGN`과 일치하는 첫 항목을 반환했고, 그 값이 결과의 `case.doc_id`에 기록됐다. 비교에 쓰는 값은 문서 ID뿐이므로 질문과 본문의 의미 유사도를 계산하지 않는다. 일치하는 항목이 없으면 `LookupError`가 발생한다.

