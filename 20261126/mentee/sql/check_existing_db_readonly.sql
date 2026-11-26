-- Day29: 오직 11/20의 report_agent_lab에서 실행. CREATE/DROP/UPDATE/DELETE 없음.
-- pgAdmin에서 이미 존재하는 report_agent_lab DB Query Tool을 사용하세요.
SELECT current_database() AS db, current_user AS role_name,
       EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector') AS vector_enabled;
SELECT category_code, count(*) AS count FROM report_read_view GROUP BY category_code ORDER BY category_code;
SELECT index_status, count(*) AS count FROM reports GROUP BY index_status ORDER BY index_status;
SELECT e.model_name,count(*) AS vectors, min(vector_dims(e.embedding)) AS dims,
       count(*) FILTER (WHERE e.indexed_revision=r.revision
                AND e.overview_hash=encode(sha256(convert_to(r.overview,'UTF8')),'hex')
                AND r.index_status='ready') AS fresh
 FROM report_embeddings e JOIN reports r USING(report_id)
 GROUP BY e.model_name;
SELECT r.report_id,v.title,v.category_code,r.index_status,r.revision
FROM reports r JOIN report_read_view v USING(report_id)
WHERE r.report_id IN (1,25,63,66,168,180) ORDER BY r.report_id;
