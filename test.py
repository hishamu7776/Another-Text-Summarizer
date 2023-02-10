'''
    summarizers = ['freq', 'artex']
    normalizers = ['raw','stem','lemma','ultra_1','ultra_2']
    document_indexes = random.sample(range(len(texts)), 50)
    index_list = []
    summarizer_list = []
    normalizer_list = []
    ROUGE_1 = []
    ROGUE_2 = []
    ROGUE_L = []
    time_list = []
    for summarizer in summarizers:
        for normalizer in normalizers:
            for idx in document_indexes:
                document = texts[idx]
                summary = summaries[idx]
                start_time = time.time()
                generated_summary = text_summarizer(document=document,
                            stopwords=stopwords,
                            min_freq=min_freq,
                            summarizer=summarizer,
                            num_sentences=num_sentences,
                            word_normalizer=normalizer)
                time_list.append(time.time()-start_time)
                r_1, r_2, r_l = Evaluator.compute_rouge(summary, generated_summary)
                index_list.append(idx)
                summarizer_list.append(summarizer)
                normalizer_list.append(normalizer)
                ROUGE_1.append(r_1)
                ROGUE_2.append(r_2)
                ROGUE_L.append(r_l)
    results = pd.DataFrame({
        'document_index':index_list,
        'summarizer':summarizer_list,
        'normalizer':normalizer_list,
        'rouge_1':ROUGE_1,
        'rouge_2':ROGUE_2,
        'rouge_l':ROGUE_L,
        'time_elapsed':time_list
    })
    # Save the dataframe as a csv file
    results.to_csv('ets_results.csv', index=False)
'''                            

