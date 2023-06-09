news_scores = {'YES': 1, 'NO': -1, 'UNKNOWN': 0}
news_ratings = ['UNKNOWN','NO','YES','YES', 'UNKNOWN', 'NO', 'YES', 'YES']
scores_sum = sum(news_scores[rating] for rating in news_ratings)
average_score = scores_sum / len(news_ratings)
print(average_score)