import re

class SQLRuleOptimizer:
    """
    Expert SQL Query Analyzer & Optimizer providing rule-based transformations,
    performance warnings, and clean formatting.
    """

    SQL_KEYWORDS = [
        "SELECT", "FROM", "WHERE", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", 
        "FULL JOIN", "ON", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET", 
        "INSERT INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE TABLE", "DROP TABLE", 
        "ALTER TABLE", "UNION", "UNION ALL", "AS", "DISTINCT", "AND", "OR", "NOT", "IN", 
        "BETWEEN", "LIKE", "IS NULL", "IS NOT NULL", "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END"
    ]

    @classmethod
    def format_sql(cls, query: str) -> str:
        """Beautify and capitalize SQL keywords."""
        sql = query.strip()
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)

        # Capitalize known keywords safely using word boundaries
        for kw in sorted(cls.SQL_KEYWORDS, key=len, reverse=True):
            pattern = re.compile(rf'\b{re.escape(kw)}\b', re.IGNORECASE)
            sql = pattern.sub(kw, sql)

        # Break lines on major clauses
        clauses = ["SELECT", "FROM", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "JOIN", 
                   "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]
        for c in clauses:
            pattern = re.compile(rf'\s*\b({re.escape(c)})\b\s*', re.IGNORECASE)
            sql = pattern.sub(r'\n\1 ', sql)

        # Clean leading newlines
        return sql.strip()

    @classmethod
    def optimize(cls, query: str, rule: str = "") -> dict:
        original = query.strip()
        formatted = cls.format_sql(original)
        optimized = formatted
        explanations = []
        rules_applied = []
        risk_level = "LOW"
        tips = []

        lower_query = original.lower()

        # Rule 1: SELECT * optimization
        if re.search(r'\bselect\s+\*\s+from\b', lower_query):
            explanations.append("Replaced 'SELECT *' recommendation: Explicit column selection minimizes network overhead and allows covering indexes.")
            rules_applied.append("SELECT_EXPLICIT_COLUMNS")
            # Suggest explicit columns in comments
            optimized = re.sub(
                r'\bSELECT\s+\*\s+FROM\b',
                'SELECT /* TODO: Replace * with required columns (e.g. id, name, created_at) */ * FROM',
                optimized,
                flags=re.IGNORECASE
            )
            tips.append("Avoid 'SELECT *' in production queries to utilize index-only scans and decrease I/O.")

        # Rule 2: Subquery IN to EXISTS or JOIN
        subquery_in = re.search(r'where\s+([\w\.]+)\s+in\s*\(\s*select\s+([\w\.]+)\s+from\s+([\w]+)\s*(?:where\s+([^)]+))?\)', lower_query)
        if subquery_in:
            col, sub_col, sub_tbl, sub_where = subquery_in.groups()
            rules_applied.append("CONVERT_IN_SUBQUERY_TO_EXISTS")
            explanations.append(f"Subquery with 'IN' on '{col}' can cause table scans; recommend 'EXISTS' or 'INNER JOIN' for index probing.")
            tips.append("EXISTS stops evaluation as soon as a match is found, whereas IN evaluates the entire subquery result set.")

        # Rule 3: Leading wildcard LIKE '%...'
        if re.search(r"like\s+['\"]%[^'\"]+['\"]", lower_query):
            rules_applied.append("AVOID_LEADING_WILDCARD_LIKE")
            explanations.append("Leading wildcard LIKE query (e.g., '%term') disables B-tree index lookups, forcing full table scan.")
            tips.append("Consider full-text search (FTS), pg_trgm gin indexes, or Elasticsearch for substring queries.")
            risk_level = "HIGH"

        # Rule 4: Redundant DISTINCT with GROUP BY
        if "distinct" in lower_query and "group by" in lower_query:
            rules_applied.append("REDUNDANT_DISTINCT_WITH_GROUP_BY")
            explanations.append("'DISTINCT' is redundant when 'GROUP BY' is present and creates duplicate sort/hash steps.")
            optimized = re.sub(r'\bSELECT\s+DISTINCT\b', 'SELECT', optimized, flags=re.IGNORECASE)

        # Rule 5: Missing WHERE filter / full table risk
        if "select" in lower_query and "from" in lower_query and "where" not in lower_query and "limit" not in lower_query:
            rules_applied.append("ADD_SAFETY_LIMIT")
            explanations.append("Query lacks a WHERE filter or LIMIT clause; added safety LIMIT 100 to prevent buffer cache pollution.")
            optimized += "\nLIMIT 100;"
            risk_level = "MEDIUM"

        # Rule 6: NOT IN on nullable column
        if re.search(r'\bnot\s+in\s*\(', lower_query):
            rules_applied.append("REPLACE_NOT_IN_WITH_NOT_EXISTS")
            explanations.append("'NOT IN' against subqueries with nullable values returns NULL/empty set and performs poorly. Use 'NOT EXISTS'.")
            tips.append("Switch 'WHERE id NOT IN (SELECT ...)' to 'WHERE NOT EXISTS (SELECT 1 FROM ... WHERE ...)' for null safety.")

        # Rule 7: User-specified rule enforcement
        if rule:
            rule_clean = rule.strip().lower()
            explanations.append(f"Applied user constraint / optimization hint: '{rule}'")
            if "index" in rule_clean:
                tips.append("Ensure indexed columns are not wrapped in functions (e.g. YEAR(date) -> use date range instead).")
            elif "join" in rule_clean:
                tips.append("Verify join predicates connect primary keys and foreign keys for hash/merge join eligibility.")

        if not explanations:
            explanations.append("Query adheres to standard indexing and execution conventions. Keyword formatting and structure aligned.")

        return {
            "original_query": original,
            "optimized_query": optimized,
            "formatted_query": formatted,
            "rules_applied": rules_applied,
            "explanations": explanations,
            "tips": tips,
            "risk_level": risk_level
        }
