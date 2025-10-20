ct_crr_query = """
SELECT *
FROM (
SELECT a.*
,b.v5_meddra_soc mapped_soc
,b.v5_ctcae_term mapped_term
,b.v5_grade_or_definition mapped_grade
,b.v5_specific mapped_specific
FROM ct_crr a

LEFT JOIN v4_grade_to_v5_grade b
ON a.ae_term=b.v4_ctcae_term
AND a.ae_grade=b.v4_grade_or_definition

WHERE ctcae_version = 4) c
WHERE mapped_term IS NOT NULL

UNION ALL

SELECT *
FROM (
SELECT a.*
,e.meddra_soc mapped_soc
,a.ae_term AS mapped_term
,a.ae_grade mapped_grade
,NULL mapped_specific
FROM ct_crr a

LEFT JOIN (SELECT * FROM ctcae_v5 WHERE grade_num=1) e
ON a.ae_term=e.ctcae_term

WHERE ctcae_version = 5
) c
"""


ct_epic_query = """
SELECT *
FROM (
SELECT a.*
,e.meddra_soc mapped_soc
,a.ae_term AS mapped_term
,a.ae_grade mapped_grade
,NULL mapped_specific

,5 AS detected_ctcae_version

FROM ct_epic a

LEFT JOIN (SELECT * FROM ctcae_v5 WHERE grade_num=1) e
ON a.ae_term=e.ctcae_term

WHERE e.meddra_soc IS NOT NULL
) c

UNION ALL

SELECT *
FROM (
SELECT a.*
,b.v5_meddra_soc mapped_soc
,b.v5_ctcae_term mapped_term
,b.v5_grade_or_definition mapped_grade
,b.v5_specific mapped_specific

,4 AS detected_ctcae_version

FROM ct_epic a

LEFT JOIN v4_grade_to_v5_grade b
ON a.ae_term=b.v4_ctcae_term
AND a.ae_grade=b.v4_grade_or_definition

LEFT JOIN (SELECT * FROM ctcae_v5 WHERE grade_num=1) e
ON a.ae_term=e.ctcae_term

WHERE e.meddra_soc IS NULL
) c
"""

