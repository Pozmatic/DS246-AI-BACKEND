# main.py
from agent import answer_legal_query
 
while True:
    query = input(">>>>>>>>>>>>>>>>>>>>>>>>>")
    ans = answer_legal_query(query)
    print(ans)
 