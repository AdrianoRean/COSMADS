def selectOperator(operator):
    if operator == "EQUAL":
        return "="
    elif operator == "GREATER":
        return ">"
    elif operator == "GREATER OR EQUAL":
        return ">="
    elif operator == "MINOR":
        return "<"
    elif operator == "MINOR OR EQUAL":
        return "<="
    elif operator == "LIKE":
        return "LIKE"