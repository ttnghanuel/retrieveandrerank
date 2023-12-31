from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import semantic_search
from torch import cuda
from flask import Flask, request
import numpy as np
import json
import py_vncorenlp

save_dir = r'D:\Nghanatuel\Sopho_Sem\NCKH\Chatbot\RetrievenRerank\VnCoreNLP'
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir = save_dir)

app = Flask(__name__)


embed_name = 'bkai-foundation-models/vietnamese-bi-encoder'
rerank_name = 'datnguyen/Cross_encoder_Zalo'
embed_model = SentenceTransformer(embed_name)
rerank_model = CrossEncoder(rerank_name)

# corpus, top_k, top_n
corpus = [
    'Đại học Bách khoa Hà Nội đã phát triển các chương trình đào tạo bằng tiếng Anh để làm cho việc học tại đây dễ dàng hơn cho sinh viên quốc tế.',
    'Môi trường học tập đa dạng và sự hỗ trợ đầy đủ cho sinh viên quốc tế tại Đại học Bách khoa Hà Nội giúp họ thích nghi nhanh chóng.',
    'Hà Nội có khí hậu mát mẻ vào mùa thu.',
    'Các món ăn ở Hà Nội rất ngon và đa dạng.'
]
threadhold = 0.5
top_k = 2
top_n = 1

# @title Retrieve and Rerank
def word_segment(text):
    result = []
    if isinstance(text, list):
        for i in text:
            result.extend(rdrsegmenter.word_segment(i))
    else:
        result = rdrsegmenter.word_segment(text)
    return result

def retrieve_and_rerank(queries, corpus, embed_model, rerank_model, threadhold, top_k, top_n):
  # tokenize
  queries = word_segment(queries)
  corpus = word_segment(corpus)

  # embed
  embed_query = embed_model.encode(queries)
  embed_corpus = embed_model.encode(corpus)

  # semantic_search
  search_results = semantic_search(embed_query, embed_corpus, top_k = top_k)
  answers = []
  for result_list in search_results:
    answer = []
    for result in result_list:
      result_dict = {}
      corpus_id = result['corpus_id']
      score = result['score']
      if score < threadhold:
        continue
      sentence = corpus[corpus_id]
      result_dict = {"corpus_id": corpus_id, "sentence": sentence, "score": score }
      answer.append(result_dict)
    answers.append(answer)

  # rerank
  rerank_results = []
  for query, result_list in zip(queries, answers):
    rerank_result = []
    for result in result_list:
      score = rerank_model.predict([query, result['sentence']])
      rerank_result.append({
          'corpus_id': result['corpus_id'],
          'score': score
          })
    rerank_results.append(rerank_result)

  # sort_rerank_result
  sorted_results = []
  for rerank_result in rerank_results:
    sorted_result = sorted(rerank_result, key=lambda x: x['score'], reverse=True)[:top_n]
    sorted_results.append(sorted_result)

  # get answer
  finalanswers = []
  for query, result_list in zip(queries, sorted_results):
    answer = []
    for result in result_list:
      score = result['score']
      corpus_id = result['corpus_id']
      sent = corpus[corpus_id]
      answer.append({
          'query': query,
          'corpus_id': corpus_id,
          'sent': sent,
          'score': score
      })
    finalanswers.append(answer)
  return finalanswers

@app.route("/query", methods=['POST'])
def get_answer():
    data = request.get_json()
    query = data.get("query", "")
    answers = retrieve_and_rerank(query, corpus, embed_model, rerank_model, threadhold, top_k, top_n)

    # Convert float32 values to float
    for answer_list in answers:
        for answer in answer_list:
            answer['score'] = float(answer['score'])

    return answers  

app.run(host = "0.0.0.0", port = 5000)
