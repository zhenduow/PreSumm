from sklearn.metrics import ndcg_score
from collections import Counter

original_lines = open('logs/abs_bert_cnndm.148000.candidate').readlines()
grammar_lines = open('logs/abs_bert_cnndm_grammar.148000.candidate').readlines()
syntax_lines = open('logs/abs_bert_cnndm_syntax.148000.candidate').readlines()
semantic_lines = open('logs/abs_bert_cnndm_semantic.148000.candidate').readlines()
lead3_lines = open('logs/abs_bert_cnndm_lead3.148000.candidate').readlines()
irrelevant_lines = open('logs/abs_bert_cnndm_irrelevant.148000.candidate').readlines()


topn = min(len(original_lines), len(grammar_lines), len(syntax_lines), len(lead3_lines), len(semantic_lines), len(irrelevant_lines))
original_score, grammar_score, syntax_score, semantic_score,lead3_score, irrelevant_score = [], [], [], [], [], []

print("BertSum")
print(topn)

for l in range(topn):
    if l%2:
        original_score.append(float(original_lines[l]))
        grammar_score.append(float(grammar_lines[l]))
        syntax_score.append(float(syntax_lines[l]))
        semantic_score.append(float(semantic_lines[l]))
        lead3_score.append(float(lead3_lines[l]))
        irrelevant_score.append(float(irrelevant_lines[l]))
            
print("original-irrelevant")
ndcg_total = ndcg_score([[1,0]]*len(original_score),list(zip(original_score, irrelevant_score)), k=1)
print(ndcg_total)

print("original-lead3")
count = 0
for k in range(len(original_score)):
    count += 1 if original_score[k] > lead3_score[k] else 0
ndcg_total = ndcg_score([[1,0]]*len(original_score),list(zip(original_score, lead3_score)), k=1)
print(ndcg_total)
#print(count, topn/2, 2*count/topn)


print("original-lead3-irrelevant")
ndcg_total = ndcg_score([[1,0,0]]*len(original_score),list(zip(original_score,lead3_score, irrelevant_score)), k=1)
print(ndcg_total)

print("original-grammar-syntax-semantic")
ndcg_total = ndcg_score([[1,0,0,0]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score)), k =1)
print(ndcg_total)

print("original-grammar-syntax-semantic-irrelevant")
ndcg_total = ndcg_score([[1,0,0,0,0]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score, irrelevant_score)), k=1)
print(ndcg_total)

